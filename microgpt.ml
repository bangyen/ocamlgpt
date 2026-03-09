(* 
   microgpt.ml - A single-file, zero-dependency GPT-2 implementation in OCaml.
   Architecture: 
   - Autograd: Scalar-valued engine with reverse-mode differentiation.
   - Model: GPT-2 with RMSNorm, Multi-Head Attention, and GELU-like ReLU MLP.
   - Optimizer: Adam with linear learning rate decay.
*)

(* --- Autograd Engine --- *)
module Value = struct
  (* A node in the computational graph. 
     _prev: parents of this node.
     _op_grad: local partial derivatives with respect to each parent. *)
  type t = {
    mutable data : float;
    mutable grad : float;
    _prev : t list;
    _op_grad : float list;
    mutable _visited : bool;
  }

  let create ?(prev=[]) ?(op_grad=[]) data =
    { data; grad = 0.0; _prev = prev; _op_grad = op_grad; _visited = false }

  (* Core Operations: define the forward value and the local gradient. *)
  let add a b = create ~prev:[a; b] ~op_grad:[1.0; 1.0] (a.data +. b.data)
  let mul a b = create ~prev:[a; b] ~op_grad:[b.data; a.data] (a.data *. b.data)
  let pow v n = create ~prev:[v] ~op_grad:[n *. (v.data ** (n -. 1.0))] (v.data ** n)
  let log v   = create ~prev:[v] ~op_grad:[1.0 /. v.data] (log v.data)
  let exp v   = let e = exp v.data in create ~prev:[v] ~op_grad:[e] e
  let relu v  = create ~prev:[v] ~op_grad:[if v.data > 0.0 then 1.0 else 0.0] (max 0.0 v.data)

  (* Derived Operations: built from core primitives. *)
  let neg v   = mul v (create (-1.0))
  let sub a b = add a (neg b)
  let div a b = mul a (pow b (-1.0))

  (* Reverse-mode Autograd: topological sort + gradient accumulation. *)
  let backward root =
    let topo = ref [] in
    let rec build v =
      if not v._visited then (v._visited <- true; List.iter build v._prev; topo := v :: !topo)
    in
    build root;
    root.grad <- 1.0;
    List.fold_left (fun () v ->
      List.iter2 (fun child og -> child.grad <- child.grad +. (og *. v.grad)) v._prev v._op_grad;
      v._visited <- false (* Reset for next step *)
    ) () !topo
end

let ( +: ), ( -: ), ( *: ), ( /: ) = Value.add, Value.sub, Value.mul, Value.div

(* --- Configuration (Hardcoded for Parity) --- *)
let n_layer = 1 (* Number of transformer blocks *)
let n_embd = 16 (* Embedding dimension *)
let block_size = 16 (* Maximum sequence length *)
let n_head = 4 (* Number of attention heads *)
let head_dim = n_embd / n_head (* Each head processes a slice of the embedding dimension. *)
let learning_rate = 0.01
let beta1 = 0.85
let beta2 = 0.99
let eps_adam = 1e-8
let num_steps = 1000

(* --- Model State --- *)
type layer = {
  wq : Value.t array array;
  wk : Value.t array array;
  wv : Value.t array array;
  wo : Value.t array array;
  fc1 : Value.t array array;
  fc2 : Value.t array array;
}

type state = { 
  wte : Value.t array array; 
  wpe : Value.t array array; 
  lm_head : Value.t array array; 
  layers : layer array; 
}

(* Global Vocabulary State *)
let vocab_size = ref 0
let uchars = ref [||]
let bos_token = ref 0

(* --- Initialization & Matrix Ops --- *)
let gauss mean std =
  let u1 = Random.float 1.0 in
  let u2 = Random.float 1.0 in
  mean +. std *. sqrt (-2.0 *. log u1) *. cos (2.0 *. Float.pi *. u2)

let matrix rows cols std =
  Array.init rows (fun _ -> Array.init cols (fun _ -> Value.create (gauss 0.0 std)))

(* Linear Layer: y = xW^T (no bias, as per microgpt.py) *)
let linear x w =
  Array.map (fun row ->
    let acc = ref (Value.create 0.0) in
    for i = 0 to Array.length x - 1 do acc := !acc +: (x.(i) *: row.(i)) done;
    !acc
  ) w

(* Softmax: exp(xi) / sum(exp(xj)). Log-sum-exp trick used for stability. *)
let softmax logits =
  let max_val = ref (-. infinity) in
  Array.iter (fun (v:Value.t) -> if v.data > !max_val then max_val := v.data) logits;
  let exps = Array.map (fun v -> Value.exp (v -: Value.create !max_val)) logits in
  let total = ref (Value.create 0.0) in
  Array.iter (fun e -> total := !total +: e) exps;
  Array.map (fun e -> e /: !total) exps

(* RMSNorm: x / sqrt(mean(x^2) + eps). A simplified LayerNorm. *)
let rmsnorm x =
  let n = float_of_int (Array.length x) in
  let ms = ref (Value.create 0.0) in
  Array.iter (fun xi -> ms := !ms +: (xi *: xi)) x;
  let avg_ms = !ms /: Value.create n in
  let scale = Value.pow (avg_ms +: Value.create 1e-5) (-0.5) in
  Array.map (fun xi -> xi *: scale) x

(* GPT Forward Pass: Multi-head attention + MLP residual blocks. *)
let gpt state token_id pos_id keys values =
  let x = Array.map2 (+:) state.wte.(token_id) state.wpe.(pos_id) |> rmsnorm in

  let rec apply_layers x li =
    if li = n_layer then x
    else
      let l = state.layers.(li) in
      (* 1) Multi-head Attention with Cache (keys/values lists) *)
      let x_norm = rmsnorm x in
      let q, k, v = linear x_norm l.wq, linear x_norm l.wk, linear x_norm l.wv in
      keys.(li) <- keys.(li) @ [k];
      values.(li) <- values.(li) @ [v];

      let x_attn = Array.make n_embd (Value.create 0.0) in
      for h = 0 to n_head - 1 do
        let hs = h * head_dim in
        let q_h = Array.sub q hs head_dim in
        let k_h = List.map (fun ki -> Array.sub ki hs head_dim) keys.(li) in
        let v_h = List.map (fun vi -> Array.sub vi hs head_dim) values.(li) in

        (* Attention scores: dot(q, k) / sqrt(dim) *)
        let attn_logits = Array.of_list (List.map (fun (kh:Value.t array) ->
          let acc = ref (Value.create 0.0) in
          for i = 0 to head_dim - 1 do acc := !acc +: (q_h.(i) *: kh.(i)) done;
          !acc /: Value.create (sqrt (float_of_int head_dim))
        ) k_h) in
        let attn_weights = softmax attn_logits in
        
        (* Weighted sum of values *)
        for j = 0 to head_dim - 1 do
          let head_out_j = ref (Value.create 0.0) in
          List.iteri (fun idx (v_h_node:Value.t array) ->
            head_out_j := !head_out_j +: (attn_weights.(idx) *: v_h_node.(j))
          ) v_h;
          x_attn.(hs + j) <- !head_out_j
        done
      done;
      let x = Array.map2 (+:) x (linear x_attn l.wo) in

      (* 2) MLP block: x = x + FC2(ReLU(FC1(norm(x)))) *)
      let x_norm_mlp = rmsnorm x in
      let mlp_act = Array.map Value.relu (linear x_norm_mlp l.fc1) in
      let mlp_out = linear mlp_act l.fc2 in
      apply_layers (Array.map2 (+:) x mlp_out) (li + 1)
  in
  linear (apply_layers x 0) state.lm_head

(* --- Main Execution --- *)

let main () =
  let open Value in
  Random.self_init ();
  (* 1. Load Data (Minimalist) *)
  if not (Sys.file_exists "input.txt") then begin
    Printf.printf "input.txt not found, downloading...\n%!";
    let _ = Sys.command "curl -s https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt" in
    ()
  end;
  let docs =
    let ic = open_in "input.txt" in
    let rec read_lines acc =
      try let line = input_line ic in read_lines (if line <> "" then line :: acc else acc)
    with End_of_file -> close_in ic; acc
    in Array.of_list (read_lines [])
  in
  let all_chars = Array.fold_left (fun s doc -> s ^ doc) "" docs |> String.to_seq |> List.of_seq |> List.sort_uniq Char.compare in
  uchars := Array.of_list all_chars;
  vocab_size := Array.length !uchars + 1;
  bos_token := Array.length !uchars;

  Printf.printf "num docs: %d\n" (Array.length docs);
  Printf.printf "vocab size: %d\n" !vocab_size;

  (* 2. Initialize Model *)
  let state = {
    wte = matrix !vocab_size n_embd 0.08;
    wpe = matrix block_size n_embd 0.08;
    lm_head = matrix !vocab_size n_embd 0.08;
    layers = Array.init n_layer (fun _ -> {
      wq = matrix n_embd n_embd 0.08;
      wk = matrix n_embd n_embd 0.08;
      wv = matrix n_embd n_embd 0.08;
      wo = matrix n_embd n_embd 0.08;
      fc1 = matrix (4 * n_embd) n_embd 0.08;
      fc2 = matrix n_embd (4 * n_embd) 0.08;
    });
  } in

  let collect_params s =
    let flatten m = Array.to_list m |> List.concat_map Array.to_list in
    let layer_params l = List.concat_map flatten [l.wq; l.wk; l.wv; l.wo; l.fc1; l.fc2] in
    List.concat [flatten s.wte; flatten s.wpe; flatten s.lm_head; 
                 List.concat_map layer_params (Array.to_list s.layers)]
  in
  let params_arr = Array.of_list (collect_params state) in
  Printf.printf "num params: %d\n" (Array.length params_arr);

  (* 3. Training Loop *)
  Random.init 42;
  let m = Array.make (Array.length params_arr) 0.0 in
  let v = Array.make (Array.length params_arr) 0.0 in

  let docs_shuffled = 
    let d = Array.to_list docs in
    let shuffle l = 
      let a = Array.of_list l in
      for i = Array.length a - 1 downto 1 do
        let j = Random.int (i + 1) in
        let temp = a.(i) in a.(i) <- a.(j); a.(j) <- temp
      done; Array.to_list a
    in Array.of_list (shuffle d)
  in

  for step = 0 to num_steps - 1 do
    let doc = docs_shuffled.(step mod Array.length docs_shuffled) in
    let tokens = [!bos_token] @ (String.to_seq doc |> Seq.map (fun c ->
      let idx = ref 0 in
      while !idx < Array.length !uchars && !uchars.(!idx) <> c do incr idx done;
      !idx
    ) |> List.of_seq) @ [!bos_token] in
    let n = min block_size (List.length tokens - 1) in

    let keys, values = Array.make n_layer [], Array.make n_layer [] in
    let losses = ref [] in

    for pos_id = 0 to n - 1 do
      let token_id, target_id = List.nth tokens pos_id, List.nth tokens (pos_id + 1) in
      let logits = gpt state token_id pos_id keys values in
      losses := Value.neg (Value.log (softmax logits).(target_id)) :: !losses
    done;
    let avg_loss = (List.fold_left (+:) (Value.create 0.0) !losses) /: (Value.create (float_of_int n)) in

    Value.backward avg_loss;

    let lr_t = learning_rate *. (1.0 -. (float_of_int step /. float_of_int num_steps)) in
    Array.iteri (fun i (p : Value.t) ->
      m.(i) <- beta1 *. m.(i) +. (1.0 -. beta1) *. p.grad;
      v.(i) <- beta2 *. v.(i) +. (1.0 -. beta2) *. (p.grad ** 2.0);
      let m_hat = m.(i) /. (1.0 -. (beta1 ** float_of_int (step + 1))) in
      let v_hat = v.(i) /. (1.0 -. (beta2 ** float_of_int (step + 1))) in
      p.data <- p.data -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam);
      p.grad <- 0.0
    ) params_arr;

    Printf.printf "step %4d / %4d | loss %.4f\r%!" (step + 1) num_steps avg_loss.data
  done;

  (* 4. Inference *)
  let temperature = 0.5 in
  Printf.printf "\n--- inference (new, hallucinated names) ---\n";
  for sample_idx = 1 to 20 do
    let keys, values = Array.make n_layer [], Array.make n_layer [] in
    let token_id = ref !bos_token in
    let sample = ref [] in

    let rec generate pos_id =
      if pos_id >= block_size then ()
      else
        let logits = gpt state !token_id pos_id keys values in
        let scaled_logits = Array.map (fun (v:Value.t) -> Value.create (v.data /. temperature)) logits in
        let probs = softmax scaled_logits in
        let r = Random.float 1.0 in
        let cumulative_prob = ref 0.0 in
        let selected_idx = ref !bos_token in
        let found = ref false in
        Array.iteri (fun i (p:Value.t) ->
          if not !found then begin
            cumulative_prob := !cumulative_prob +. p.data;
            if r <= !cumulative_prob then (selected_idx := i; found := true)
          end
        ) probs;
        token_id := !selected_idx;
        if !token_id <> !bos_token then (sample := !sample @ [!uchars.(!token_id)]; generate (pos_id + 1))
    in
    generate 0;
    Printf.printf "sample %2d: %s\n" sample_idx (String.of_seq (List.to_seq !sample))
  done

let () =
  if Array.length Sys.argv > 0 && (Filename.basename Sys.argv.(0) = "microgpt.ml" || Filename.basename Sys.argv.(0) = "microgpt") then
    main ()
