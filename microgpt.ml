(* 
   microgpt.ml - A single-file, zero-dependency GPT-2 implementation in OCaml.
   Architecture: 
   - Autograd: Scalar-valued engine with reverse-mode differentiation.
   - Model: GPT-2 with RMSNorm, Multi-Head Attention, and GELU-like ReLU MLP.
   - Optimizer: Adam with linear learning rate decay.
*)

(* --- Scalar Autograd Engine --- 
   This module tracks a computation graph of scalar values to compute gradients via reverse-mode AD.
*)
module Value = struct
  type t = {
    mutable data     : float;
    mutable grad     : float;
    _prev            : t list;     (* Ancestor nodes in the computation graph *)
    _op_grad         : float list; (* Local derivatives w.r.t. each ancestor *)
    mutable _visited : bool;
  }

  let create data prev op_grad =
    { data; grad = 0.0; _prev = prev; _op_grad = op_grad; _visited = false }
  let scalar d = create d [] []
  let zero = scalar 0.0

  (* Basic operations: each creates a new node and stores local gradients (Chain Rule) *)
  let add a b = create (a.data +. b.data) [a; b] [1.0; 1.0]
  let mul a b = create (a.data *. b.data) [a; b] [b.data; a.data]
  let pow v n = create (v.data ** n) [v] [n *. (v.data ** (n -. 1.0))]
  let log v   = create (log v.data) [v] [1.0 /. v.data]
  let exp v   = let e = exp v.data in create e [v] [e]

  let relu v  =
    create (max 0.0 v.data) [v]
    [if v.data > 0.0 then 1.0 else 0.0]

  let neg v   = mul v (scalar (-1.0))
  let div a b = mul a (pow b (-1.0))
  let sub a b = add a (neg b)

  (* Reverse-mode autodiff: traverse the graph in reverse topological order *)
  let backward root =
    let rec build v topo =
      if v._visited then topo
      else begin
        v._visited <- true;
        let topo' = List.fold_left (fun acc child -> build child acc) topo v._prev in
        v :: topo'
      end
    in
    let topo = build root [] in
    root.grad <- 1.0; (* Seed the gradient of the loss with 1.0 *)
    List.iter (fun v ->
      (* Propagate gradients to children using the stored chain rule derivatives *)
      List.iter2 (fun child og -> 
        child.grad <- child.grad +. (og *. v.grad)
      ) v._prev v._op_grad;
      v._visited <- false
    ) topo

  let data v = v.data
  let grad v = v.grad
  let set_data v d = v.data <- d
  let set_grad v g = v.grad <- g
end

(* Operator aliases for convenience *)
let ( +: ), ( -: ), ( *: ), ( /: ) = Value.add, Value.sub, Value.mul, Value.div
let ( +^ ) = Array.map2 (+:)

(* --- Configuration --- *)
let n_layer       = 1               (* Number of transformer blocks *)
let n_embd        = 16              (* Embedding dimension *)
let block_size    = 16              (* Maximum sequence length *)
let n_head        = 4               (* Number of attention heads *)
let head_dim      = n_embd / n_head (* Each head processes a slice of the embedding dimension. *)
let learning_rate = 0.01
let beta1         = 0.85
let beta2         = 0.99
let eps_adam      = 1e-8
let num_steps     = 1000

(* --- Model State --- *)
type layer = {
  wq  : Value.t array array;
  wk  : Value.t array array;
  wv  : Value.t array array;
  wo  : Value.t array array;
  fc1 : Value.t array array;
  fc2 : Value.t array array;
}

type state = { 
  wte : Value.t array array; 
  wpe : Value.t array array; 
  lm_head : Value.t array array; 
  layers : layer array; 
}

(* --- Global Configuration & Vocabulary --- *)
let () = Random.init 42

let docs = 
  if not (Sys.file_exists "input.txt") then
    ignore (Sys.command "curl -s https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt");
  In_channel.with_open_text "input.txt" In_channel.input_lines
  |> List.filter ((<>) "") 
  |> Array.of_list

let uchars = 
  String.concat "" (Array.to_list docs)
  |> String.to_seq |> List.of_seq 
  |> List.sort_uniq Char.compare 
  |> Array.of_list

let vocab_size = Array.length uchars + 1
let bos_token = Array.length uchars

(* --- Initialization & Matrix Ops --- *)

(* Box-Muller transform for Gaussian sampling *)
let gauss mean std =
  let u1 = Random.float 1.0 in
  let u2 = Random.float 1.0 in
  mean +. std *. sqrt (-2.0 *. log u1) *. cos (2.0 *. Float.pi *. u2)

let matrix ?(std = 0.08) rows cols =
  Array.init rows (fun _ -> 
    Array.init cols (fun _ -> Value.scalar (gauss 0.0 std))
  )

(* Dot Product: a . b *)
let dot a b =
  Seq.fold_left2 (fun acc va vb -> acc +: (va *: vb))
    Value.zero (Array.to_seq a) (Array.to_seq b)

(* Linear Layer: y = xW^T (Arguments flipped for pipelining) *)
let linear w x = Array.map (dot x) w

(* Softmax: exp(xi) / sum(exp(xj)) *)
let softmax logits =
  let max_val = Array.fold_left (fun m v -> max m (Value.data v)) (-.infinity) logits in
  let exps = Array.map (fun v -> Value.exp (v -: Value.scalar max_val)) logits in
  let total = Array.fold_left (+:) Value.zero exps in
  Array.map (fun e -> e /: total) exps

(* ReLU Activation: max(0, x) *)
let relu = Array.map Value.relu

(* RMSNorm: x / sqrt(mean(x^2) + eps) *)
let rmsnorm x =
  let ms    = dot x x /: Value.scalar (float (Array.length x)) in
  let scale = Value.pow (ms +: Value.scalar 1e-5) (-0.5) in
  Array.map (fun xi -> xi *: scale) x

(* --- GPT Forward Pass --- *)
let gpt state token_id pos_id keys values =
  let x = state.wte.(token_id) +^ state.wpe.(pos_id) |> rmsnorm in

  let rec apply_layers x li =
    if li = n_layer then x
    else
      let l = state.layers.(li) in
      let x_norm = x |> rmsnorm in
      let q, k, v = x_norm |> linear l.wq, x_norm |> linear l.wk, x_norm |> linear l.wv in
      
      (* KV Caching: Store previous K/V for autoregressive inference/context *)
      keys.(li) <- keys.(li) @ [k];
      values.(li) <- values.(li) @ [v];

      (* Multi-Head Attention *)
      let x_attn = 
        List.init n_head (fun h ->
          let hs = h * head_dim in
          let q_h = Array.sub q hs head_dim in
          let k_h = List.map (fun ki -> Array.sub ki hs head_dim) keys.(li) in
          let v_h = List.map (fun vi -> Array.sub vi hs head_dim) values.(li) in
          
          let attn_weights = 
            k_h |> List.map (fun kh ->
              dot q_h kh /: Value.scalar (sqrt (float head_dim))
            ) |> Array.of_list |> softmax
          in
          
          Array.init head_dim (fun hj ->
            List.fold_left2 (fun a weight vhi ->
              a +: (weight *: vhi.(hj))
            ) Value.zero (Array.to_list attn_weights) v_h
          )
        ) |> Array.concat
      in
      
      (* Residual Connection + FFN *)
      let x = x_attn |> linear l.wo |> ( +^ ) x in
      let mlp_out = 
        x |> rmsnorm 
        |> linear l.fc1 
        |> relu 
        |> linear l.fc2
      in
      apply_layers (x +^ mlp_out) (li + 1)
  in
  apply_layers x 0 |> linear state.lm_head

(* --- Main Execution --- *)

let main () =
  Printf.printf "num docs: %d\n" (Array.length docs);
  Printf.printf "vocab size: %d\n" vocab_size;

  (* 2. Initialize Model *)
  let state = {
    wte     = matrix vocab_size n_embd;
    wpe     = matrix block_size n_embd;
    lm_head = matrix vocab_size n_embd;
    layers  = Array.init n_layer (fun _ -> {
      wq  = matrix n_embd n_embd;
      wk  = matrix n_embd n_embd;
      wv  = matrix n_embd n_embd;
      wo  = matrix n_embd n_embd;
      fc1 = matrix (4 * n_embd) n_embd;
      fc2 = matrix n_embd (4 * n_embd);
    });
  } in

  let collect_params s =
    let flatten m = Array.to_list m |> List.concat_map Array.to_list in
    let layer_params l = List.concat_map flatten [l.wq; l.wk; l.wv; l.wo; l.fc1; l.fc2] in
    List.concat [
      flatten s.wte; 
      flatten s.wpe; 
      flatten s.lm_head; 
      List.concat_map layer_params (Array.to_list s.layers)
    ]
  in
  
  let params_arr = Array.of_list (collect_params state) in
  Printf.printf "num params: %d\n" (Array.length params_arr);

  (* 3. Training Loop *)
  let m = Array.make (Array.length params_arr) 0.0 in
  let v = Array.make (Array.length params_arr) 0.0 in

  let docs_shuffled = 
    let a = Array.copy docs in
    for i = Array.length a - 1 downto 1 do
      let j = Random.int (i + 1) in
      let t = a.(i) in a.(i) <- a.(j); a.(j) <- t
    done; 
    a
  in

  let rec train_loop step =
    if step >= num_steps then ()
    else begin
      let doc = docs_shuffled.(step mod Array.length docs_shuffled) in
      let tokens = 
        [bos_token] @ (String.to_seq doc |> Seq.map (fun c ->
          let rec find i = if uchars.(i) = c then i else find (i + 1) in find 0
        ) |> List.of_seq) @ [bos_token] 
      in
      let n = min block_size (List.length tokens - 1) in

      let keys   = Array.make n_layer [] in
      let values = Array.make n_layer [] in

      let losses = 
        List.init n (fun pos_id ->
          let token_id  = List.nth tokens pos_id in
          let target_id = List.nth tokens (pos_id + 1) in
          let logits    = gpt state token_id pos_id keys values in
          logits |> softmax |> fun p -> Value.log p.(target_id) |> Value.neg
        )
      in
      
      let total_loss = List.fold_left (+:) Value.zero losses in
      let avg_loss = total_loss /: (Value.scalar (float_of_int n)) in

      Value.backward avg_loss;

      (* Adam Optimizer update with linear learning rate decay *)
      let lr_t = learning_rate *. (1.0 -. (float_of_int step /. float_of_int num_steps)) in
      Array.iteri (fun i p ->
        m.(i) <- beta1 *. m.(i) +. (1.0 -. beta1) *. Value.grad p;
        v.(i) <- beta2 *. v.(i) +. (1.0 -. beta2) *. (Value.grad p ** 2.0);
        let m_hat = m.(i) /. (1.0 -. (beta1 ** float_of_int (step + 1))) in
        let v_hat = v.(i) /. (1.0 -. (beta2 ** float_of_int (step + 1))) in
        let delta = lr_t *. m_hat /. (sqrt v_hat +. eps_adam) in
        Value.set_data p (Value.data p -. delta);
        Value.set_grad p 0.0
      ) params_arr;

      Printf.printf
        "step %4d / %4d | loss %.4f\r%!"
        (step + 1) num_steps (Value.data avg_loss);
      
      train_loop (step + 1)
    end
  in
  train_loop 0;

  (* 4. Inference *)
  let temperature = 0.5 in
  Printf.printf "\n--- inference (new, hallucinated names) ---\n";
  let rec infer_loop sample_idx =
    if sample_idx > 20 then ()
    else begin
      let keys     = Array.make n_layer [] in
      let values   = Array.make n_layer [] in
      let rec generate pos_id tokens =
        if pos_id >= block_size then tokens
        else
          let token_id = List.hd (List.rev tokens) in
          let logits = gpt state token_id pos_id keys values in
          let scaled_logits = 
            Array.map (fun v -> Value.scalar (Value.data v /. temperature)) logits 
          in
          let probs = softmax scaled_logits in
          let r = Random.float 1.0 in
          let rec sample_prob i cum =
            if i >= vocab_size then bos_token else
            let cum = cum +. Value.data probs.(i) in
            if r <= cum then i else sample_prob (i + 1) cum
          in
          let next_id = sample_prob 0 0.0 in
          if next_id = bos_token then tokens
          else generate (pos_id + 1) (tokens @ [next_id])
      in
      let sample_ids = generate 0 [bos_token] |> List.tl in
      let sample_chars = List.map (fun id -> uchars.(id)) sample_ids in
      let sample_str = String.of_seq (List.to_seq sample_chars) in
      Printf.printf "sample %2d: %s\n" sample_idx sample_str;
      infer_loop (sample_idx + 1)
    end
  in
  infer_loop 1

let () = main ()
