(* microgpt.ml - OCaml port of Karpathy's microgpt.py *)

(* --- Autograd Engine --- *)
module Value = struct
  type t = {
    mutable data : float;
    mutable grad : float;
    _children : t list;
    _local_grads : float list;
    mutable _visited : bool;
  }

  let create ?(children=[]) ?(local_grads=[]) data =
    { data; grad = 0.0; _children = children; _local_grads = local_grads; _visited = false }

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

  let backward root =
    let topo = ref [] in
    let rec build_topo v =
      if not v._visited then begin
        v._visited <- true;
        List.iter build_topo v._children;
        topo := v :: !topo
      end
    in
    build_topo root;
    root.grad <- 1.0;
    List.iter (fun v ->
      List.iter2 (fun child local_grad ->
        child.grad <- child.grad +. (local_grad *. v.grad)
      ) v._children v._local_grads;
      v._visited <- false (* Reset visited for next step *)
    ) !topo
end

(* Helper operators for Value *)
let ( +: ) = Value.add
let ( -: ) = Value.sub
let ( *: ) = Value.mul
let ( /: ) = Value.div

(* --- Configuration --- *)
let n_layer = 4
let n_embd = 48
let block_size = 32
let n_head = 4
let head_dim = n_embd / n_head
let learning_rate = 0.001
let beta1 = 0.9
let beta2 = 0.999
let eps_adam = 1e-8
let num_steps = 5000

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
  Array.map (fun row ->
    let acc = ref (Value.create 0.0) in
    for i = 0 to Array.length x - 1 do
      acc := !acc +: (x.(i) *: row.(i))
    done;
    !acc
  ) w

let softmax logits =
  let max_val = ref (-. infinity) in
  Array.iter (fun (v:Value.t) -> if v.data > !max_val then max_val := v.data) logits;
  let exps = Array.map (fun v -> Value.exp (v -: Value.create !max_val)) logits in
  let total = ref (Value.create 0.0) in
  Array.iter (fun e -> total := !total +: e) exps;
  Array.map (fun e -> e /: !total) exps

let rmsnorm x =
  let n = float_of_int (Array.length x) in
  let ms = ref (Value.create 0.0) in
  Array.iter (fun xi -> ms := !ms +: (xi *: xi)) x;
  let avg_ms = !ms /: Value.create n in
  let scale = Value.pow (avg_ms +: Value.create 1e-5) (-0.5) in
  Array.map (fun xi -> xi *: scale) x

let gpt state token_id pos_id keys values =
  let tok_emb = state.wte.(token_id) in
  let pos_emb = state.wpe.(pos_id) in
  let x = Array.map2 (+:) tok_emb pos_emb |> rmsnorm in

  let rec apply_layers x li =
    if li = n_layer then x
    else
      let layer = state.layers.(li) in
      (* 1) Multi-head Attention block *)
      let x_norm_attn = rmsnorm x in
      let q = linear x_norm_attn (Hashtbl.find layer "attn_wq") in
      let k = linear x_norm_attn (Hashtbl.find layer "attn_wk") in
      let v = linear x_norm_attn (Hashtbl.find layer "attn_wv") in
      keys.(li) <- keys.(li) @ [k];
      values.(li) <- values.(li) @ [v];

      let x_attn = Array.make n_embd (Value.create 0.0) in
      for h = 0 to n_head - 1 do
        let hs = h * head_dim in
        let q_h = Array.sub q hs head_dim in
        let k_h = List.map (fun ki -> Array.sub ki hs head_dim) keys.(li) in
        let v_h = List.map (fun vi -> Array.sub vi hs head_dim) values.(li) in

        let attn_logits = Array.of_list (List.map (fun (kh:Value.t array) ->
          let acc = ref (Value.create 0.0) in
          for i = 0 to head_dim - 1 do
            acc := !acc +: (q_h.(i) *: kh.(i))
          done;
          !acc /: Value.create (sqrt (float_of_int head_dim))
        ) k_h) in
        let attn_weights = softmax attn_logits in
        
        for j = 0 to head_dim - 1 do
          let head_out_j = ref (Value.create 0.0) in
          List.iteri (fun idx (v_h_node:Value.t array) ->
            head_out_j := !head_out_j +: (attn_weights.(idx) *: v_h_node.(j))
          ) v_h;
          x_attn.(hs + j) <- !head_out_j
        done
      done;
      let x_attn_out = linear x_attn (Hashtbl.find layer "attn_wo") in
      let x = Array.map2 (+:) x x_attn_out in

      (* 2) MLP block *)
      let x_norm_mlp = rmsnorm x in
      let mlp_fc1 = linear x_norm_mlp (Hashtbl.find layer "mlp_fc1") in
      let mlp_act = Array.map Value.relu mlp_fc1 in
      let mlp_fc2 = linear mlp_act (Hashtbl.find layer "mlp_fc2") in
      let x = Array.map2 (+:) x mlp_fc2 in
      apply_layers x (li + 1)
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
    layers = Array.init n_layer (fun _ ->
      let h = Hashtbl.create 6 in
      Hashtbl.add h "attn_wq" (matrix n_embd n_embd 0.08);
      Hashtbl.add h "attn_wk" (matrix n_embd n_embd 0.08);
      Hashtbl.add h "attn_wv" (matrix n_embd n_embd 0.08);
      Hashtbl.add h "attn_wo" (matrix n_embd n_embd 0.08);
      Hashtbl.add h "mlp_fc1" (matrix (4 * n_embd) n_embd 0.08);
      Hashtbl.add h "mlp_fc2" (matrix n_embd (4 * n_embd) 0.08);
      h
    );
  } in

  let params_list = ref [] in
  Array.iter (fun row -> Array.iter (fun p -> params_list := p :: !params_list) row) state.wte;
  Array.iter (fun row -> Array.iter (fun p -> params_list := p :: !params_list) row) state.wpe;
  Array.iter (fun row -> Array.iter (fun p -> params_list := p :: !params_list) row) state.lm_head;
  Array.iter (fun layer ->
    let keys = ["attn_wq"; "attn_wk"; "attn_wv"; "attn_wo"; "mlp_fc1"; "mlp_fc2"] in
    List.iter (fun k ->
      let mat = Hashtbl.find layer k in
      Array.iter (fun row -> Array.iter (fun p -> params_list := p :: !params_list) row) mat
    ) keys
  ) state.layers;
  let params_arr = Array.of_list (List.rev !params_list) in
  Printf.printf "num params: %d\n" (Array.length params_arr);

  (* 3. Training Loop *)
  Random.init 42;
  let m = Array.make (Array.length params_arr) 0.0 in
  let v = Array.make (Array.length params_arr) 0.0 in

  (* Shuffling *)
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

    let keys = Array.make n_layer [] in
    let values = Array.make n_layer [] in
    let losses = ref [] in

    for pos_id = 0 to n - 1 do
      let token_id = List.nth tokens pos_id in
      let target_id = List.nth tokens (pos_id + 1) in
      let logits = gpt state token_id pos_id keys values in
      let probs = softmax logits in
      let loss_t = Value.neg (Value.log probs.(target_id)) in
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
      let m_hat = m.(i) /. (1.0 -. (beta1 ** float_of_int (step + 1))) in
      let v_hat = v.(i) /. (1.0 -. (beta2 ** float_of_int (step + 1))) in
      p.data <- p.data -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam);
      p.grad <- 0.0
    ) params_arr;

    if step mod 10 = 0 || step = 0 then
      Printf.printf "step %4d / %d | loss %.4f\n%!" (step + 1) num_steps avg_loss.data
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
        let probs = softmax logits in
        let r = Random.float 1.0 in
        let cumulative_prob = ref 0.0 in
        let selected_idx = ref (!bos_token) in
        let found = ref false in
        Array.iteri (fun i (p:Value.t) ->
          if not !found then begin
            cumulative_prob := !cumulative_prob +. p.data;
            if r <= !cumulative_prob then (selected_idx := i; found := true)
          end
        ) probs;
        token_id := !selected_idx;
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
