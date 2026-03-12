(*
   microgpt.ml - A single-file, zero-dependency GPT-2 implementation in OCaml.

   Architecture:
   - Autograd: Scalar-valued engine with reverse-mode differentiation.
   - Model: GPT-2 with RMSNorm, Multi-Head Attention, and GELU-like ReLU MLP.
   - Optimizer: Adam with linear learning rate decay.
*)

(* --- Scalar Autograd Engine --- *)
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
        let inner = fun acc child -> build child acc in
        let topo' = List.fold_left inner topo v._prev in
        v :: topo'
      end
    in
    let topo = build root [] in
    (* Seed the gradient of the loss with 1.0 *)
    root.grad <- 1.0;
    (* Propagate gradients to children using the stored chain rule derivatives *)
    List.iter (fun v ->
      List.iter2 (fun child og ->
        child.grad <-
          child.grad +. (og *. v.grad)
      ) v._prev v._op_grad;
      v._visited <- false
    ) topo

  let data v = v.data
  let grad v = v.grad
  let set_data v d = v.data <- d
  let set_grad v g = v.grad <- g
end

(* --- Operator Aliases --- *)
let ( +: ), ( -: ) = Value.add, Value.sub
let ( *: ), ( /: ) = Value.mul, Value.div
let ( +^ ), ( *^ ) = Array.map2 (+:), Array.map2 ( *:)
let sum = Array.fold_left (+:) Value.zero

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
type mat = Value.t array array

type layer = {
  wq  : mat;
  wk  : mat;
  wv  : mat;
  wo  : mat;
  fc1 : mat;
  fc2 : mat;
}

type state = {
  wte     : mat;
  wpe     : mat;
  lm_head : mat;
  layers  : layer array;
}

(* --- Data Loading & Vocabulary --- *)
let () = Random.init 42

let docs =
  if not (Sys.file_exists "input.txt") then
    ignore (Sys.command "curl -s https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt");
  In_channel.with_open_text "input.txt" (fun ic ->
    In_channel.input_all ic |> String.split_on_char '\n')
  |> List.filter ((<>) "")
  |> Array.of_list

let uchars =
  String.concat "" (Array.to_list docs)
  |> String.to_seq |> List.of_seq
  |> List.sort_uniq Char.compare
  |> Array.of_list

let vocab_size = Array.length uchars + 1
let bos_token = Array.length uchars

(* --- Initialization & Utils --- *)

(** gauss: Box-Muller transform for Gaussian sampling. *)
let gauss mean std =
  let u1 = Random.float 1.0 in
  let u2 = Random.float 1.0 in
  mean +. std
    *. sqrt (-2.0 *. log u1)
    *. cos (2.0 *. Float.pi *. u2)

(** matrix: Initializes a matrix (2D array) of scalar values with Gaussian weights. *)
let matrix ?(std = 0.08) rows cols =
  Array.init rows (fun _ ->
    Array.init cols (fun _ -> Value.scalar (gauss 0.0 std))
  )

(** dot: Dot Product - sum(a * b). *)
let dot a b = a *^ b |> sum

(** linear: Linear Layer - y = xW^T (Arguments flipped for pipelining). *)
let linear w x = Array.map (dot x) w

(** softmax: Normalizes logits into a probability distribution. *)
let softmax logits =
  let max_val = Array.fold_left (fun m v -> max m (Value.data v)) (-.infinity) logits in
  let exps = Array.map (fun v -> Value.exp (v -: Value.scalar max_val)) logits in
  Array.map (fun e -> e /: sum exps) exps

(** rmsnorm: Root Mean Square Layer Normalization. *)
let rmsnorm x =
  let ms    = dot x x /: Value.scalar (float (Array.length x)) in
  let scale = Value.pow (ms +: Value.scalar 1e-5) (-0.5) in
  Array.map (fun xi -> xi *: scale) x

(* --- GPT Forward Pass --- *)

(** gpt: GPT forward pass for a single token at a given position. *)
let gpt state token_id pos_id keys values =
  let x = state.wte.(token_id)
    +^ state.wpe.(pos_id)
    |> rmsnorm in

  let rec apply_layers x li =
    if li = n_layer then x
    else
      let l = state.layers.(li) in
      let x_norm = x |> rmsnorm in
      let q, k, v =
        x_norm |> linear l.wq,
        x_norm |> linear l.wk,
        x_norm |> linear l.wv in

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
              dot q_h kh /:
                Value.scalar
                (sqrt (float head_dim))
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
      let x = x_attn
        |> linear l.wo
        |> ( +^ ) x in
      let mlp_out = x
        |> rmsnorm
        |> linear l.fc1
        |> Array.map Value.relu
        |> linear l.fc2
      in
      apply_layers
        (x +^ mlp_out) (li + 1)
  in
  apply_layers x 0
    |> linear state.lm_head

(* --- Model Utilities --- *)

(** cross_entropy: Computes negative log likelihood loss. *)
let cross_entropy logits target_id =
  logits
    |> softmax
    |> fun p -> Value.log p.(target_id)
    |> Value.neg

(** zip_loss: Maps over a sequence of tokens to compute the cross-entropy loss at each step. *)
let zip_loss state keys values =
  let rec loop pos_id = function
    | t1 :: t2 :: ts when pos_id < block_size ->
        let l = cross_entropy
          (gpt state t1 pos_id keys values) t2 in
        l :: loop (pos_id + 1) (t2 :: ts)
    | _ -> []
  in
  loop 0

(** sample: Picks an index i with probability probs.(i). *)
let sample probs =
  let r = Random.float 1.0 in

  let rec loop i cum =
    let cum' = cum +. Value.data probs.(i) in
    if r <= cum' || i = Array.length probs - 1
    then i else loop (i + 1) cum'
  in loop 0 0.0

(** scale: Scale logits by temperature and detach from computation graph. *)
let scale temp logits =
  Array.map (fun v -> Value.scalar
    (Value.data v /. temp)) logits

(** predict: Predict the next token ID given current position and token ID. *)
let predict state temp keys values pos tid =
  gpt state tid pos keys values
    |> scale temp
    |> softmax
    |> sample

(** generate: Produces a sequence of token IDs using autoregressive sampling. *)
let generate state temperature keys values =
  let next = predict state temperature keys values in
  (0, bos_token)
  |> Seq.unfold (fun (pos, tid) ->
      if pos < block_size then
        match next pos tid with
        | id when id = bos_token -> None
        | id -> Some (id, (pos + 1, id))
      else None)
  |> List.of_seq

(* --- Training Ops --- *)

(** step_adam: Performs an Adam optimization step. *)
let step_adam params m v step =
  let lr_t =
    learning_rate *. (1.0 -.
      (float_of_int step /. float_of_int num_steps))
  in
  let b1_t = 1.0 -. (beta1 ** float_of_int (step + 1)) in
  let b2_t = 1.0 -. (beta2 ** float_of_int (step + 1)) in
  let inv_b1 = 1.0 -. beta1 in
  let inv_b2 = 1.0 -. beta2 in

  Array.iteri
    (fun i p ->
      let g = Value.grad p in
      m.(i) <- (beta1 *. m.(i)) +. (inv_b1 *. g);
      v.(i) <- (beta2 *. v.(i)) +. (inv_b2 *. (g *. g));

      let m_hat = m.(i) /. b1_t in
      let v_hat = v.(i) /. b2_t in
      let delta = lr_t *. m_hat /.
        (sqrt v_hat +. eps_adam) in

      Value.set_data p
        (Value.data p -. delta);
      Value.set_grad p 0.0)
    params

(* --- Main Execution --- *)

(** main: Initializes model, runs training loop, and performs inference. *)

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
    let flatten m =
      Array.to_list m |> List.concat_map Array.to_list in
    let layer_params l = List.concat_map flatten
      [l.wq; l.wk; l.wv; l.wo; l.fc1; l.fc2]
    in
    List.concat [
      flatten s.wte;
      flatten s.wpe;
      flatten s.lm_head;
      List.concat_map
        layer_params
        (Array.to_list s.layers)
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
      let t = a.(i) in
      let j = Random.int (i + 1) in
        a.(i) <- a.(j); a.(j) <- t
    done;
    a
  in

  let rec train_loop step =
    if step < num_steps then begin
      let ind = step mod
        Array.length docs_shuffled in
      let doc = docs_shuffled.(ind) in

      let tokens =
        bos_token
        :: (String.to_seq doc
           |> Seq.map (fun c ->
                let rec find i =
                  if uchars.(i) = c then
                    i else find (i + 1)
                in
                find 0)
           |> List.of_seq)
        @ [ bos_token ]
      in

      let keys = Array.make n_layer [] in
      let values = Array.make n_layer [] in

      let losses = zip_loss state keys values tokens in
      let total_loss = losses |> Array.of_list |> sum in
      let number = List.length losses in
  
      let avg_loss =
        total_loss /: Value.scalar
        (float_of_int number) in

      Value.backward avg_loss;
      step_adam params_arr m v step;

      Printf.printf "step %4d / %4d | loss %.4f\r%!" 
        (step + 1) num_steps (Value.data avg_loss);

      train_loop (step + 1)
    end
  in
  train_loop 0;

  (* 4. Inference *)
  let temperature = 0.5 in
  Printf.printf "\n--- inference (new, hallucinated names) ---\n";

  let rec infer_loop sample_idx =
    if sample_idx <= 20 then begin
      let keys = Array.make n_layer [] in
      let values = Array.make n_layer [] in

      let sample_ids = generate state temperature keys values in
      let sample_chars = List.map (fun id -> uchars.(id)) sample_ids in
      let sample_str = String.of_seq (List.to_seq sample_chars) in

      Printf.printf "sample %2d: %s\n" sample_idx sample_str;
      infer_loop (sample_idx + 1)
    end
  in
  infer_loop 1

let () = main ()
