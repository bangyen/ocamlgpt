(* 
   microgpt.ml - A single-file, zero-dependency GPT-2 implementation in OCaml.
   Architecture: 
   - Autograd: Scalar-valued engine with reverse-mode differentiation.
   - Model: GPT-2 with RMSNorm, Multi-Head Attention, and GELU-like ReLU MLP.
   - Optimizer: Adam with linear learning rate decay.
*)

module type ENGINE = sig
  type t
  val create : ?prev:t list -> ?op_grad:float list -> float -> t
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val neg : t -> t
  val pow : t -> float -> t
  val exp : t -> t
  val log : t -> t
  val relu : t -> t
  val backward : t -> unit
  val data : t -> float
  val grad : t -> float
  val set_data : t -> float -> unit
  val set_grad : t -> float -> unit
end

(* --- Scalar Autograd Engine --- 
   This module tracks a computation graph of scalar values to compute gradients via reverse-mode AD.
*)
module Value : ENGINE = struct
  type t = {
    mutable data : float;
    mutable grad : float;
    _prev : t list;       (* Ancestor nodes in the computation graph *)
    _op_grad : float list; (* Local derivatives w.r.t. each ancestor *)
    mutable _visited : bool;
  }

  let create ?(prev=[]) ?(op_grad=[]) data =
    { data; grad = 0.0; _prev = prev; _op_grad = op_grad; _visited = false }

  (* Basic operations: each creates a new node and stores local gradients (Chain Rule) *)
  let add a b = create ~prev:[a; b] ~op_grad:[1.0; 1.0] (a.data +. b.data)
  let mul a b = create ~prev:[a; b] ~op_grad:[b.data; a.data] (a.data *. b.data)
  let pow v n = create ~prev:[v] ~op_grad:[n *. (v.data ** (n -. 1.0))] (v.data ** n)
  let log v   = create ~prev:[v] ~op_grad:[1.0 /. v.data] (log v.data)
  let exp v   = let e = exp v.data in create ~prev:[v] ~op_grad:[e] e
  let relu v  = create ~prev:[v] ~op_grad:[if v.data > 0.0 then 1.0 else 0.0] (max 0.0 v.data)

  let neg v   = mul v (create (-1.0))
  let sub a b = add a (neg b)
  let div a b = mul a (pow b (-1.0))

  (* Reverse-mode autodiff: traverse the graph in reverse topological order *)
  let backward root =
    let topo = ref [] in
    let rec build v =
      if not v._visited then (v._visited <- true; List.iter build v._prev; topo := v :: !topo)
    in
    build root;
    root.grad <- 1.0; (* Seed the gradient of the loss with 1.0 *)
    List.iter (fun v ->
      (* Propagate gradients to children using the stored chain rule derivatives *)
      List.iter2 (fun child og -> child.grad <- child.grad +. (og *. v.grad)) v._prev v._op_grad;
      v._visited <- false
    ) !topo

  let data v = v.data
  let grad v = v.grad
  let set_data v d = v.data <- d
  let set_grad v g = v.grad <- g
end

let ( +: ), ( -: ), ( *: ), ( /: ) = Value.add, Value.sub, Value.mul, Value.div

(* --- Configuration --- *)
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

(* Linear Layer: y = xW^T *)
let linear x w =
  w |> Array.map (fun row ->
    let acc = ref (Value.create 0.0) in
    for i = 0 to Array.length x - 1 do acc := !acc +: (x.(i) *: row.(i)) done;
    !acc
  )

(* Softmax: exp(xi) / sum(exp(xj)) *)
let softmax logits =
  let max_val = Array.fold_left (fun m (v:Value.t) -> max m (Value.data v)) (-. infinity) logits in
  let exps = logits |> Array.map (fun v -> Value.exp (v -: Value.create max_val)) in
  let total = Array.fold_left (+:) (Value.create 0.0) exps in
  exps |> Array.map (fun e -> e /: total)

(* RMSNorm: x / sqrt(mean(x^2) + eps) *)
let rmsnorm x =
  let n = float_of_int (Array.length x) in
  let ms = Array.fold_left (fun acc xi -> acc +: (xi *: xi)) (Value.create 0.0) x in
  let scale = (ms /: Value.create n) +: Value.create 1e-5 |> fun v -> Value.pow v (-0.5) in
  x |> Array.map (fun xi -> xi *: scale)

(* --- GPT Forward Pass --- *)
let gpt state token_id pos_id keys values =
  let x = Array.map2 (+:) state.wte.(token_id) state.wpe.(pos_id) |> rmsnorm in

  let rec apply_layers x li =
    if li = n_layer then x
    else
      let l = state.layers.(li) in
      let x_norm = x |> rmsnorm in
      let q, k, v = linear x_norm l.wq, linear x_norm l.wk, linear x_norm l.wv in
      (* KV Caching: Store previous K/V for autoregressive inference/context *)
      keys.(li) <- keys.(li) @ [k];
      values.(li) <- values.(li) @ [v];

      let x_attn = Array.init n_embd (fun j ->
        let h = j / head_dim in
        let hj = j mod head_dim in
        let hs = h * head_dim in
        let q_h = Array.sub q hs head_dim in
        let k_h = List.map (fun ki -> Array.sub ki hs head_dim) keys.(li) in
        let v_h = List.map (fun vi -> Array.sub vi hs head_dim) values.(li) in

        let attn_weights = 
          k_h |> List.map (fun kh ->
            let acc = ref (Value.create 0.0) in
            for i = 0 to head_dim - 1 do acc := !acc +: (q_h.(i) *: kh.(i)) done;
            !acc /: Value.create (sqrt (float_of_int head_dim))
          ) |> Array.of_list |> softmax
        in
        
        let acc = ref (Value.create 0.0) in
        List.iteri (fun i vhi -> acc := !acc +: (attn_weights.(i) *: vhi.(hj))) v_h;
        !acc
      ) in
      
      let x = Array.map2 (+:) x (linear x_attn l.wo) in
      let x_norm_mlp = x |> rmsnorm in
      let mlp_out = 
        linear x_norm_mlp l.fc1 
        |> Array.map Value.relu 
        |> fun act -> linear act l.fc2
      in
      apply_layers (Array.map2 (+:) x mlp_out) (li + 1)
  in
  apply_layers x 0 |> fun out -> linear out state.lm_head

(* --- Main Execution --- *)

let main () =
  Random.init 42;
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
    in Array.of_list (List.rev (read_lines []))
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
  (* 3. Training Loop *)
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
      losses := (logits |> softmax |> fun p -> Value.log p.(target_id) |> Value.neg) :: !losses
    done;
    let avg_loss = (!losses |> List.fold_left (+:) (Value.create 0.0)) /: (Value.create (float_of_int n)) in

    Value.backward avg_loss;

    let lr_t = learning_rate *. (1.0 -. (float_of_int step /. float_of_int num_steps)) in
    Array.iteri (fun i (p : Value.t) ->
      m.(i) <- beta1 *. m.(i) +. (1.0 -. beta1) *. Value.grad p;
      v.(i) <- beta2 *. v.(i) +. (1.0 -. beta2) *. (Value.grad p ** 2.0);
      let m_hat = m.(i) /. (1.0 -. (beta1 ** float_of_int (step + 1))) in
      let v_hat = v.(i) /. (1.0 -. (beta2 ** float_of_int (step + 1))) in
      Value.set_data p (Value.data p -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam));
      Value.set_grad p 0.0
    ) params_arr;

    Printf.printf "step %4d / %4d | loss %.4f\r%!" (step + 1) num_steps (Value.data avg_loss)
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
        let scaled_logits = Array.map (fun (v:Value.t) -> Value.create (Value.data v /. temperature)) logits in
        let probs = softmax scaled_logits in
        let r = Random.float 1.0 in
        let cumulative_prob = ref 0.0 in
        let selected_idx = ref !bos_token in
        let found = ref false in
        Array.iteri (fun i (p:Value.t) ->
          if not !found then begin
            cumulative_prob := !cumulative_prob +. Value.data p;
            if r <= !cumulative_prob then (selected_idx := i; found := true)
          end
        ) probs;
        token_id := !selected_idx;
        if !token_id <> !bos_token then (sample := !sample @ [!uchars.(!token_id)]; generate (pos_id + 1))
    in
    generate 0;
    Printf.printf "sample %2d: %s\n" sample_idx (String.of_seq (List.to_seq !sample))
  done

let () = main ()
