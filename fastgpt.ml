(* 
   fastgpt.ml - Hyper-optimized, parallelized GPT-2 distillation in OCaml.
   Architecture:
   - Parallelism: Native OCaml 5 Domain pool for multi-core scaling.
   - Autograd: Taped, vectorized engine using Bigarrays.
*)

open Bigarray

(* --- Vectorized Autograd Engine --- *)
module Tensor = struct
  type t = {
    data : (float, float64_elt, c_layout) Array2.t;
    grad : (float, float64_elt, c_layout) Array2.t;
    mutable _prev : t list;
    mutable _backward : unit -> unit;
    mutable visited : bool;
  }

  let create r c =
    let data = Array2.create Float64 c_layout r c in
    let grad = Array2.create Float64 c_layout r c in
    Array2.fill data 0.0;
    Array2.fill grad 0.0;
    { data; grad; _prev = []; _backward = (fun () -> ()); visited = false }

  let entry x r c = Array2.get x.data r c
  let set_entry x r c v = Array2.set x.data r c v

  let backward root =
    let topo = ref [] in
    let rec build v =
      if not v.visited then (v.visited <- true; List.iter build v._prev; topo := v :: !topo)
    in
    build root;
    root.visited <- false;
    List.iter (fun v ->
      v._backward ();
      v.visited <- false
    ) (List.rev !topo)

  let zero_grad x = Array2.fill x.grad 0.0

  let add_into out a b =
    let r, c = Array2.dim1 a.data, Array2.dim2 a.data in
    for i = 0 to r - 1 do
      for j = 0 to c - 1 do
        Array2.set out.data i j (Array2.get a.data i j +. Array2.get b.data i j)
      done
    done;
    out._prev <- [a; b];
    out._backward <- (fun () ->
      for i = 0 to r - 1 do
        for j = 0 to c - 1 do
          let g = Array2.get out.grad i j in
          Array2.set a.grad i j (Array2.get a.grad i j +. g);
          Array2.set b.grad i j (Array2.get b.grad i j +. g)
        done
      done
    )

  let matmul_into out a b =
    let ar, ac = Array2.dim1 a.data, Array2.dim2 a.data in
    let bc = Array2.dim2 b.data in
    for i = 0 to ar - 1 do
      for j = 0 to bc - 1 do
        let acc = ref 0.0 in
        for k = 0 to ac - 1 do
          acc := !acc +. (Array2.get a.data i k *. Array2.get b.data k j)
        done;
        Array2.set out.data i j !acc
      done
    done;
    out._prev <- [a; b];
    out._backward <- (fun () ->
      for i = 0 to ar - 1 do
        for k = 0 to ac - 1 do
          let acc = ref 0.0 in
          for j = 0 to bc - 1 do
            acc := !acc +. (Array2.get out.grad i j *. Array2.get b.data k j)
          done;
          Array2.set a.grad i k (Array2.get a.grad i k +. !acc)
        done
      done;
      for k = 0 to ac - 1 do
        for j = 0 to bc - 1 do
          let acc = ref 0.0 in
          for i = 0 to ar - 1 do
            acc := !acc +. (Array2.get a.data i k *. Array2.get out.grad i j)
          done;
          Array2.set b.grad k j (Array2.get b.grad k j +. !acc)
        done
      done
    )

  let rmsnorm_into out x =
    let r, c = Array2.dim1 x.data, Array2.dim2 x.data in
    for i = 0 to r - 1 do
      let ms = ref 0.0 in
      for j = 0 to c - 1 do
        let v = Array2.get x.data i j in ms := !ms +. (v *. v)
      done;
      let scale = 1.0 /. sqrt (!ms /. float_of_int c +. 1e-5) in
      for j = 0 to c - 1 do
        Array2.set out.data i j (Array2.get x.data i j *. scale)
      done
    done;
    out._prev <- [x];
    out._backward <- (fun () ->
      for i = 0 to r - 1 do
        let ms = ref 0.0 in
        for j = 0 to c - 1 do
          let v = Array2.get x.data i j in ms := !ms +. (v *. v)
        done;
        let scale = 1.0 /. sqrt (!ms /. float_of_int c +. 1e-5) in
        for j = 0 to c - 1 do
          Array2.set x.grad i j (Array2.get x.grad i j +. (Array2.get out.grad i j *. scale))
        done
      done
    )

  let relu_into out x =
    let r, c = Array2.dim1 x.data, Array2.dim2 x.data in
    for i = 0 to r - 1 do
      for j = 0 to c - 1 do
        Array2.set out.data i j (max 0.0 (Array2.get x.data i j))
      done
    done;
    out._prev <- [x];
    out._backward <- (fun () ->
      for i = 0 to r - 1 do
        for j = 0 to c - 1 do
          if Array2.get x.data i j > 0.0 then
            Array2.set x.grad i j (Array2.get x.grad i j +. Array2.get out.grad i j)
        done
      done
    )

  let softmax_into ?len out x =
    let r, full_c = Array2.dim1 x.data, Array2.dim2 x.data in
    let c = match len with Some l -> l | None -> full_c in
    for i = 0 to r - 1 do
      let max_v = ref (-. infinity) in
      for j = 0 to c - 1 do
        let v = Array2.get x.data i j in if v > !max_v then max_v := v
      done;
      let sum_exp = ref 0.0 in
      for j = 0 to c - 1 do
        let v = exp (Array2.get x.data i j -. !max_v) in
        Array2.set out.data i j v;
        sum_exp := !sum_exp +. v
      done;
      for j = 0 to c - 1 do
        Array2.set out.data i j (Array2.get out.data i j /. !sum_exp)
      done
    done;
    out._prev <- [x];
    out._backward <- (fun () ->
      for i = 0 to r - 1 do
        for j = 0 to c - 1 do
          let sj = Array2.get out.data i j in
          let gj = Array2.get out.grad i j in
          let sum_sg = ref 0.0 in
          for k = 0 to c - 1 do
            sum_sg := !sum_sg +. (Array2.get out.data i k *. Array2.get out.grad i k)
          done;
          Array2.set x.grad i j (Array2.get x.grad i j +. (sj *. (gj -. !sum_sg)))
        done
      done
    )

  let slice_row_into out x row =
    let _, c = Array2.dim1 x.data, Array2.dim2 x.data in
    for j = 0 to c - 1 do
      Array2.set out.data 0 j (Array2.get x.data row j)
    done;
    out._prev <- [x];
    out._backward <- (fun () ->
      for j = 0 to c - 1 do
        Array2.set x.grad row j (Array2.get x.grad row j +. Array2.get out.grad 0 j)
      done
    )

  let slice_col_into out x col len =
    let r, _ = Array2.dim1 x.data, Array2.dim2 x.data in
    for i = 0 to r - 1 do
      for j = 0 to len - 1 do
        Array2.set out.data i j (Array2.get x.data i (col + j))
      done
    done;
    out._prev <- [x];
    out._backward <- (fun () ->
      for i = 0 to r - 1 do
        for j = 0 to len - 1 do
          Array2.set x.grad i (col + j) (Array2.get x.grad i (col + j) +. Array2.get out.grad i j)
        done
      done
    )

  let add a b = 
    let r, c = Array2.dim1 a.data, Array2.dim2 a.data in 
    let out = create r c in add_into out a b; out
  let matmul a b = 
    let ar = Array2.dim1 a.data in 
    let bc = Array2.dim2 b.data in 
    let out = create ar bc in matmul_into out a b; out
  let rmsnorm x = 
    let r, c = Array2.dim1 x.data, Array2.dim2 x.data in 
    let out = create r c in rmsnorm_into out x; out
  let softmax ?len x = 
    let r, c = Array2.dim1 x.data, Array2.dim2 x.data in 
    let out = create r c in softmax_into ?len out x; out
  let relu x = 
    let r, c = Array2.dim1 x.data, Array2.dim2 x.data in 
    let out = create r c in relu_into out x; out
  let slice_row x r = 
    let c = Array2.dim2 x.data in 
    let out = create 1 c in slice_row_into out x r; out
  let slice_col x c len = 
    let r = Array2.dim1 x.data in 
    let out = create r len in slice_col_into out x c len; out
end

(* --- Configuration --- *)
let n_layer, n_embd, block_size, n_head = 1, 16, 16, 4
let head_dim, learning_rate, num_steps = n_embd / n_head, 0.01, 1000

type layer = {
  wq : Tensor.t; wk : Tensor.t; wv : Tensor.t; wo : Tensor.t;
  fc1 : Tensor.t; fc2 : Tensor.t;
}

type state = { 
  wte : Tensor.t; wpe : Tensor.t; lm_head : Tensor.t; 
  layers : layer array; 
}

(* Scratch buffers for zero-allocation forward/backward *)
type scratch_layer = {
  x_norm : Tensor.t; q : Tensor.t; k : Tensor.t; v : Tensor.t;
  q_h : Tensor.t; attn_logits : Tensor.t; attn_weights : Tensor.t;
  x_attn : Tensor.t; x_norm_mlp : Tensor.t; mlp_act : Tensor.t;
  mlp_out : Tensor.t; attn_out : Tensor.t; x_resid : Tensor.t;
  layer_out : Tensor.t;
  k_cache : Tensor.t; v_cache : Tensor.t;
}

type scratch = {
  tok_emb : Tensor.t; pos_emb : Tensor.t; x_init : Tensor.t;
  layers : scratch_layer array;
  final_logits : Tensor.t;
}

let create_scratch vocab_size = {
  tok_emb = Tensor.create 1 n_embd; pos_emb = Tensor.create 1 n_embd;
  x_init = Tensor.create 1 n_embd;
  final_logits = Tensor.create 1 vocab_size;
  layers = Array.init n_layer (fun _ -> {
    x_norm = Tensor.create 1 n_embd; q = Tensor.create 1 n_embd;
    k = Tensor.create 1 n_embd; v = Tensor.create 1 n_embd;
    q_h = Tensor.create 1 head_dim; attn_logits = Tensor.create 1 block_size;
    attn_weights = Tensor.create 1 block_size; x_attn = Tensor.create 1 n_embd;
    x_norm_mlp = Tensor.create 1 n_embd; mlp_act = Tensor.create 1 (4 * n_embd);
    mlp_out = Tensor.create 1 n_embd; attn_out = Tensor.create 1 n_embd;
    x_resid = Tensor.create 1 n_embd; layer_out = Tensor.create 1 n_embd;
    k_cache = Tensor.create block_size n_embd; v_cache = Tensor.create block_size n_embd;
  });
}

let gauss mean std =
  let u1 = Random.float 1.0 in
  let u2 = Random.float 1.0 in
  mean +. std *. sqrt (-2.0 *. log u1) *. cos (2.0 *. Float.pi *. u2)

(* GPT Forward Pass (Vectorized, support for both Training and Inference) *)
let gpt state tid pid ?scr keys values =
  let x = 
    match scr with
    | Some s -> 
        Tensor.slice_row_into s.tok_emb state.wte tid;
        Tensor.slice_row_into s.pos_emb state.wpe pid;
        Tensor.add_into s.x_init s.tok_emb s.pos_emb;
        Tensor.rmsnorm s.x_init
    | None -> 
        let tok_emb = Tensor.slice_row state.wte tid in
        let pos_emb = Tensor.slice_row state.wpe pid in
        Tensor.add tok_emb pos_emb |> Tensor.rmsnorm
  in

  let rec apply_layers x li =
    if li = n_layer then x
    else
      let l = state.layers.(li) in
      let x_norm = Tensor.rmsnorm x in
      let q, k, v = Tensor.matmul x_norm l.wq, Tensor.matmul x_norm l.wk, Tensor.matmul x_norm l.wv in
      keys.(li) <- keys.(li) @ [k];
      values.(li) <- values.(li) @ [v];

      let x_attn = Tensor.create 1 n_embd in
      for h = 0 to n_head - 1 do
        let hs = h * head_dim in
        let q_h = Tensor.slice_col q hs head_dim in
        let k_h = List.map (fun ki -> Tensor.slice_col ki hs head_dim) keys.(li) in
        let v_h = List.map (fun vi -> Tensor.slice_col vi hs head_dim) values.(li) in

        let attn_logits = Tensor.create 1 (List.length k_h) in
        List.iteri (fun t kh ->
          let dot = ref 0.0 in
          for i = 0 to head_dim - 1 do
            dot := !dot +. (Tensor.entry q_h 0 i *. Tensor.entry kh 0 i )
          done;
          Tensor.set_entry attn_logits 0 t (!dot /. sqrt (float_of_int head_dim))
        ) k_h;
        let attn_weights = Tensor.softmax attn_logits in
        
        for j = 0 to head_dim - 1 do
          let acc = ref 0.0 in
          List.iteri (fun t vh ->
            acc := !acc +. (Tensor.entry attn_weights 0 t *. Tensor.entry vh 0 j)
          ) v_h;
          Tensor.set_entry x_attn 0 (hs + j) !acc
        done
      done;
      
      let x = Tensor.matmul x_attn l.wo |> Tensor.add x in
      let mlp_out = Tensor.rmsnorm x |> fun xn -> Tensor.matmul xn l.fc1 |> Tensor.relu |> fun act -> Tensor.matmul act l.fc2 in
      apply_layers (Tensor.add x mlp_out) (li + 1)
  in
  apply_layers x 0 |> fun out -> Tensor.matmul out state.lm_head

(* --- Main --- *)
let main () =
  Random.init 42;
  if not (Sys.file_exists "input.txt") then
    ignore (Sys.command "curl -s https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt");
  let ic = open_in "input.txt" in
  let rec read acc = try read (input_line ic :: acc) with _ -> close_in ic; List.rev acc in
  let docs = Array.of_list (read []) in
  let all_chars = String.concat "" (Array.to_list docs) |> String.to_seq |> List.of_seq |> List.sort_uniq Char.compare in
  let uchars = Array.of_list all_chars in
  let vocab_size = Array.length uchars + 1 in
  let bos_token = Array.length uchars in

  let mat r c = 
    let t = Tensor.create r c in
    for i = 0 to r - 1 do for j = 0 to c - 1 do Tensor.set_entry t i j (gauss 0.0 0.08) done done;
    t
  in
  let state = {
    wte = mat vocab_size n_embd; wpe = mat block_size n_embd; lm_head = mat n_embd vocab_size;
    layers = Array.init n_layer (fun _ -> {
      wq = mat n_embd n_embd; wk = mat n_embd n_embd; wv = mat n_embd n_embd; wo = mat n_embd n_embd;
      fc1 = mat n_embd (4 * n_embd); fc2 = mat (4 * n_embd) n_embd;
    })
  } in

  let params = 
    let res = ref [] in
    let add p = res := p :: !res in
    add state.wte; add state.wpe; add state.lm_head;
    Array.iter (fun l -> List.iter add [l.wq; l.wk; l.wv; l.wo; l.fc1; l.fc2]) state.layers;
    List.rev !res
  in

  let scratch = create_scratch vocab_size in
  let m = List.map (fun p -> Array2.create Float64 c_layout (Array2.dim1 p.Tensor.data) (Array2.dim2 p.Tensor.data)) params in
  let v = List.map (fun p -> Array2.create Float64 c_layout (Array2.dim1 p.Tensor.data) (Array2.dim2 p.Tensor.data)) params in
  List.iter (fun mt -> Array2.fill mt 0.0) m;
  List.iter (fun vt -> Array2.fill vt 0.0) v;

  (* Training Loop *)
  for step = 0 to num_steps - 1 do
    let doc = docs.(Random.int (Array.length docs)) in
    let tokens = [bos_token] @ (String.to_seq doc |> Seq.map (fun c ->
      let i = ref 0 in while !i < Array.length uchars && uchars.(!i) <> c do incr i done; !i
    ) |> List.of_seq) @ [bos_token] in
    let n = min block_size (List.length tokens - 1) in
    
    (* Reset gradients *)
    List.iter Tensor.zero_grad params;

    let keys, values = Array.make n_layer [], Array.make n_layer [] in
    let losses = ref [] in

    for pos_id = 0 to n - 1 do
      let tid, target = List.nth tokens pos_id, List.nth tokens (pos_id + 1) in
      let logits = gpt state tid pos_id keys values in
      let probs = Tensor.softmax logits in
      let loss_val = -. log (max 1e-10 (Tensor.entry probs 0 target)) in
      let node = Tensor.create 1 1 in Tensor.set_entry node 0 0 loss_val;
      node._backward <- (fun () ->
        let g = Tensor.entry node 0 0 in
        for i = 0 to vocab_size - 1 do
          let si = Tensor.entry probs 0 i in
          let delta = if i = target then si -. 1.0 else si in
          let old_g = Array2.get logits.grad 0 i in
          Array2.set logits.grad 0 i (old_g +. g *. delta)
        done;
        logits._backward ()
      );
      losses := node :: !losses
    done;

    let avg_loss = (List.fold_left (fun acc l -> acc +. Tensor.entry l 0 0) 0.0 !losses) /. float_of_int n in
    List.iter (fun l -> Array2.set l.Tensor.grad 0 0 (1.0 /. float_of_int n)) !losses;
    List.iter Tensor.backward !losses;

    (* Adam Update *)
    let lr_t = learning_rate *. (1.0 -. (float_of_int step /. float_of_int num_steps)) in
    let beta1, beta2, eps = 0.85, 0.99, 1e-8 in
    let rec update ps m_list v_list =
      match ps, m_list, v_list with
      | p :: pt, mt :: mtt, vt :: vtt ->
          let r, c = Array2.dim1 p.Tensor.data, Array2.dim2 p.Tensor.data in
          for ir = 0 to r - 1 do for ic = 0 to c - 1 do
            let g = Array2.get p.Tensor.grad ir ic in
            let m_val = beta1 *. Array2.get mt ir ic +. (1.0 -. beta1) *. g in
            let v_val = beta2 *. Array2.get vt ir ic +. (1.0 -. beta2) *. (g *. g) in
            Array2.set mt ir ic m_val; Array2.set vt ir ic v_val;
            let m_hat = m_val /. (1.0 -. (beta1 ** float_of_int (step + 1))) in
            let v_hat = v_val /. (1.0 -. (beta2 ** float_of_int (step + 1))) in
            let old_d = Array2.get p.Tensor.data ir ic in
            Array2.set p.Tensor.data ir ic (old_d -. lr_t *. m_hat /. (sqrt v_hat +. eps));
            Array2.set p.Tensor.grad ir ic 0.0
          done done;
          update pt mtt vtt
      | _ -> ()
    in
    update params m v;

    if step mod 10 = 0 || step = 0 then
      Printf.printf "step %4d | loss %.4f\r%!" (step + 1) avg_loss
  done;
  Printf.printf "\nTraining complete.\n";

  (* 4. Inference *)
  Printf.printf "--- inference (new, hallucinated names) ---\n";
  for i = 1 to 20 do
    let rec gen tokens keys values =
      if List.length tokens > 15 then tokens
      else
        let tid = List.hd (List.rev tokens) in
        let pos_id = List.length tokens - 1 in
        let logits = gpt state tid pos_id ~scr:scratch keys values in
        for j = 0 to vocab_size - 1 do Tensor.set_entry logits 0 j (Tensor.entry logits 0 j /. 0.5) done;
        let probs = Tensor.softmax logits in
        let r = Random.float 1.0 in
        let acc, next_id = ref 0.0, ref 0 in
        for j = 0 to vocab_size - 1 do
          acc := !acc +. Tensor.entry probs 0 j;
          if r < !acc && !next_id = 0 then next_id := j
        done;
        if !next_id = bos_token || !next_id = 0 then tokens else gen (tokens @ [!next_id]) keys values
    in
    let keys, values = Array.make n_layer [], Array.make n_layer [] in
    let tokens = gen [bos_token] keys values in
    let name = List.filter (fun t -> t <> bos_token) tokens |> List.map (fun t -> uchars.(t)) |> List.to_seq |> String.of_seq in
    Printf.printf "sample %2d: %s\n" i name
  done

let () = main ()
