(* microgpt.ml - OCaml port of Karpathy's microgpt.py *)

(* --- Autograd Engine --- *)
module Value = struct
  type t = {
    mutable data : float;
    mutable grad : float;
    _children : t list;
    _local_grads : float list;
  }

  let create ?(children=[]) ?(local_grads=[]) data =
    { data; grad = 0.0; _children = children; _local_grads = local_grads }

  let add v1 v2 =
    create ~children:[v1; v2] ~local_grads:[1.0; 1.0] (v1.data +. v2.data)

  let mul v1 v2 =
    create ~children:[v1; v2] ~local_grads:[v2.data; v1.data] (v1.data *. v2.data)

  let pow v n =
    create ~children:[v] ~local_grads:[n *. (v.data ** (n -. 1.0))] (v.data ** n)

  let log v =
    create ~children:[v] ~local_grads:[1.0 /. v.data] (log v.data)

  let exp v =
    let e = exp v.data in
    create ~children:[v] ~local_grads:[e] e

  let relu v =
    create ~children:[v] ~local_grads:[if v.data > 0.0 then 1.0 else 0.0] (max 0.0 v.data)

  let neg v = mul v (create (-1.0))
  let sub v1 v2 = add v1 (neg v2)
  let div v1 v2 = mul v1 (pow v2 (-1.0))

  (* Topological sort for backward pass *)
  let backward root =
    let topo = ref [] in
    let visited = ref [] in
    let rec build_topo v =
      if not (List.memq v !visited) then begin
        visited := v :: !visited;
        List.iter build_topo v._children;
        topo := v :: !topo
      end
    in
    build_topo root;
    root.grad <- 1.0;
    List.iter (fun v ->
      List.iter2 (fun child local_grad ->
        child.grad <- child.grad +. (local_grad *. v.grad)
      ) v._children v._local_grads
    ) (List.rev !topo)
end

(* Helper operators for Value *)
let ( +: ) = Value.add
let ( -: ) = Value.sub
let ( *: ) = Value.mul
let ( /: ) = Value.div

(* --- Configuration --- *)
let n_layer = 1
let n_embd = 16
let block_size = 16
let n_head = 4
let head_dim = n_embd / n_head
let learning_rate = 0.01
let beta1 = 0.85
let beta2 = 0.99
let eps_adam = 1e-8
let num_steps = 1000

(* --- Random Initialization --- *)
let gauss mean std =
  let u1 = Random.float 1.0 in
  let u2 = Random.float 1.0 in
  mean +. std *. sqrt (-2.0 *. log u1) *. cos (2.0 *. Float.pi *. u2)

let matrix rows cols std =
  Array.init rows (fun _ -> Array.init cols (fun _ -> Value.create (gauss 0.0 std)))

(* --- Model State --- *)
type state = {
  wte : Value.t array array;          (* vocab_size x n_embd *)
  wpe : Value.t array array;          (* block_size x n_embd *)
  lm_head : Value.t array array;      (* vocab_size x n_embd *)
  layers : (string, Value.t array array) Hashtbl.t array;
}

(* Placeholder for vocab_size, will set after loading data *)
let vocab_size = ref 0
let uchars = ref [||]
let bos_token = ref 0

(* --- Model Forward Pass --- *)
let linear x w =
  Array.to_list (Array.map (fun row ->
    List.fold_left2 (fun acc xi wi -> acc +: (xi *: wi)) (Value.create 0.0) x (Array.to_list row)
  ) w)

let softmax logits =
  let max_val = List.fold_left (fun acc (v:Value.t) -> max acc v.data) (-. infinity) logits in
  let exps = List.map (fun v -> Value.exp (v -: Value.create max_val)) logits in
  let total = List.fold_left (+:) (Value.create 0.0) exps in
  List.map (fun e -> e /: total) exps

let rmsnorm x =
  let n = float_of_int (List.length x) in
  let ms = (List.fold_left (fun acc xi -> acc +: (xi *: xi)) (Value.create 0.0) x) /: Value.create n in
  let scale = Value.pow (ms +: Value.create 1e-5) (-0.5) in
  List.map (fun xi -> xi *: scale) x

let gpt state token_id pos_id keys values =
  let tok_emb = Array.to_list state.wte.(token_id) in
  let pos_emb = Array.to_list state.wpe.(pos_id) in
  let x = List.map2 (+:) tok_emb pos_emb |> rmsnorm in

  let rec apply_layers x li =
    if li = n_layer then x
    else
      let layer = state.layers.(li) in
      let x_norm = rmsnorm x in
      let q = linear x_norm (Hashtbl.find layer "attn_wq") in
      let k = linear x_norm (Hashtbl.find layer "attn_wk") in
      let v = linear x_norm (Hashtbl.find layer "attn_wv") in
      keys.(li) <- keys.(li) @ [k];
      values.(li) <- values.(li) @ [v];

      let x_attn = ref [] in
      for h = 0 to n_head - 1 do
        let hs = h * head_dim in
        let q_h = List.filteri (fun i _ -> i >= hs && i < hs + head_dim) q in
        let k_h = List.map (fun ki -> List.filteri (fun i _ -> i >= hs && i < hs + head_dim) ki) keys.(li) in
        let v_h = List.map (fun vi -> List.filteri (fun i _ -> i >= hs && i < hs + head_dim) vi) values.(li) in

        let attn_logits = List.map (fun kh ->
          (List.fold_left2 (fun acc qh kh_j -> acc +: (qh *: kh_j)) (Value.create 0.0) q_h kh) /: Value.create (sqrt (float_of_int head_dim))
        ) k_h in
        let attn_weights = softmax attn_logits in
        let head_out = Array.init head_dim (fun j ->
           List.fold_left2 (fun acc weight (vh:Value.t list) -> acc +: (weight *: (List.nth vh j))) (Value.create 0.0) attn_weights v_h
        ) |> Array.to_list in
        x_attn := !x_attn @ head_out
      done;
      let x_attn_out = linear !x_attn (Hashtbl.find layer "attn_wo") in
      let x_after_attn = List.map2 (+:) x x_attn_out |> rmsnorm in

      (* MLP *)
      let mlp_fc1 = linear x_after_attn (Hashtbl.find layer "mlp_fc1") in
      let mlp_act = List.map Value.relu mlp_fc1 in
      let mlp_fc2 = linear mlp_act (Hashtbl.find layer "mlp_fc2") in
      let x_after_mlp = List.map2 (+:) x_after_attn mlp_fc2 in
      apply_layers x_after_mlp (li + 1)
  in
  let x_final = apply_layers x 0 in
  linear x_final state.lm_head

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
    in read_lines []
  in
  let all_chars = List.fold_left (fun s doc -> s ^ doc) "" docs |> String.to_seq |> List.of_seq |> List.sort_uniq Char.compare in
  uchars := Array.of_list all_chars;
  vocab_size := Array.length !uchars + 1;
  bos_token := Array.length !uchars;

  Printf.printf "num docs: %d\n" (List.length docs);
  Printf.printf "vocab size: %d\n" !vocab_size;

  (* 2. Initialize Model *)
  let state = {
    wte = matrix !vocab_size n_embd 0.02;
    wpe = matrix block_size n_embd 0.02;
    lm_head = matrix !vocab_size n_embd 0.02;
    layers = Array.init n_layer (fun _ ->
      let h = Hashtbl.create 6 in
      Hashtbl.add h "attn_wq" (matrix n_embd n_embd 0.02);
      Hashtbl.add h "attn_wk" (matrix n_embd n_embd 0.02);
      Hashtbl.add h "attn_wv" (matrix n_embd n_embd 0.02);
      Hashtbl.add h "attn_wo" (matrix n_embd n_embd 0.02);
      Hashtbl.add h "mlp_fc1" (matrix (4 * n_embd) n_embd 0.02);
      Hashtbl.add h "mlp_fc2" (matrix n_embd (4 * n_embd) 0.02);
      h
    );
  } in

  let params_list = ref [] in
  Array.iter (fun row -> Array.iter (fun p -> params_list := p :: !params_list) row) state.wte;
  Array.iter (fun row -> Array.iter (fun p -> params_list := p :: !params_list) row) state.wpe;
  Array.iter (fun row -> Array.iter (fun p -> params_list := p :: !params_list) row) state.lm_head;
  Array.iter (fun layer ->
    Hashtbl.iter (fun _ mat -> Array.iter (fun row -> Array.iter (fun p -> params_list := p :: !params_list) row) mat) layer
  ) state.layers;
  let params_arr = Array.of_list (List.rev !params_list) in
  Printf.printf "num params: %d\n" (Array.length params_arr);

  (* 3. Training Loop (Single doc for simplicity, loop for N steps) *)
  let m = Array.make (Array.length params_arr) 0.0 in
  let v = Array.make (Array.length params_arr) 0.0 in

  for step = 1 to num_steps do
    let doc = List.nth docs (Random.int (List.length docs)) in
    let tokens = [!bos_token] @ (String.to_seq doc |> Seq.map (fun c ->
      let idx = ref 0 in
      while !idx < Array.length !uchars && !uchars.(!idx) <> c do incr idx done;
      !idx
    ) |> List.of_seq) @ [!bos_token] in
    let n = min block_size (List.length tokens - 1) in

    let keys = Array.make n_layer [] in
    let values = Array.make n_layer [] in
    let losses = ref [] in

    for pos_id = 0 to n - 1 do
      let token_id = List.nth tokens pos_id in
      let target_id = List.nth tokens (pos_id + 1) in
      let logits = gpt state token_id pos_id keys values in
      let probs = softmax logits in
      let loss_t = Value.neg (Value.log (List.nth probs target_id)) in
      losses := loss_t :: !losses
    done;
    let sum_loss = List.fold_left (+:) (Value.create 0.0) !losses in
    let avg_loss = sum_loss /: (Value.create (float_of_int n)) in

    (* Backward *)
    Value.backward avg_loss;

    (* Adam Update *)
    let lr_t = learning_rate *. (1.0 -. (float_of_int step /. float_of_int num_steps)) in
    Array.iteri (fun i (p : Value.t) ->
      m.(i) <- beta1 *. m.(i) +. (1.0 -. beta1) *. p.grad;
      v.(i) <- beta2 *. v.(i) +. (1.0 -. beta2) *. (p.grad ** 2.0);
      let m_hat = m.(i) /. (1.0 -. (beta1 ** float_of_int step)) in
      let v_hat = v.(i) /. (1.0 -. (beta2 ** float_of_int step)) in
      p.data <- p.data -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam);
      p.grad <- 0.0
    ) params_arr;

    if step mod 10 = 0 || step = 1 then
      Printf.printf "step %4d / %d | loss %.4f\n%!" step num_steps avg_loss.data
  done;

  (* 4. Inference *)
  Printf.printf "\n--- inference (new, hallucinated names) ---\n";
  for sample_idx = 1 to 10 do
    let keys = Array.make n_layer [] in
    let values = Array.make n_layer [] in
    let token_id = ref !bos_token in
    let sample = ref [] in

    let rec generate pos_id =
      if pos_id >= block_size then ()
      else
        let logits = gpt state !token_id pos_id keys values in
        (* Greedy decoding for simplicity in OCaml port *)
        let probs = softmax logits in
        let max_idx = ref 0 in
        let max_prob = ref (-. 1.0) in
        List.iteri (fun i (p:Value.t) -> if p.data > !max_prob then (max_prob := p.data; max_idx := i)) probs;
        token_id := !max_idx;
        if !token_id <> !bos_token then begin
          sample := !sample @ [!uchars.(!token_id)];
          generate (pos_id + 1)
        end
    in
    generate 0;
    let name = String.of_seq (List.to_seq !sample) in
    Printf.printf "sample %2d: %s\n" sample_idx name
  done

let () =
  if Array.length Sys.argv > 0 && (Filename.basename Sys.argv.(0) = "microgpt.ml" || Filename.basename Sys.argv.(0) = "microgpt") then
    main ()
