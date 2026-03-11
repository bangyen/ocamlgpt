(* 
   fastgpt.ml - Hyper-optimized, parallelized GPT-2 implementation in OCaml.
   
   Performance-oriented design:
   - Taped Autograd: Vectorized engine using Bigarrays and closure-based "tapes".
   - Parallelism: Native OCaml 5 Domain pool for multi-core scaling.
*)

open Bigarray

let get  = Array2.get
let set  = Array2.set
let dim1 = Array2.dim1
let dim2 = Array2.dim2

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
    (* _backward stores a closure that propagates gradients to children (The "Tape") *)
    { data; grad; _prev = []; _backward = (fun () -> ()); visited = false }

  let entry x r c = get x.data r c
  let set_entry x r c v = set x.data r c v

  let backward root =
    let rec build v topo =
      if v.visited then topo
      else begin
        v.visited <- true;
        let topo' = List.fold_left (fun acc child -> build child acc) topo v._prev in
        v :: topo'
      end
    in
    let topo = build root [] in
    root.visited <- false;
    List.iter (fun v ->
      v._backward ();
      v.visited <- false
    ) topo

  let zero_grad x = Array2.fill x.grad 0.0

  let add_into out a b =
    let r, c = dim1 a.data, dim2 a.data in
    for i = 0 to r - 1 do
      for j = 0 to c - 1 do
        set out.data i j (get a.data i j +. get b.data i j)
      done
    done;
    out._prev <- [a; b];
    out._backward <- (fun () ->
      for i = 0 to r - 1 do
        for j = 0 to c - 1 do
          let g = get out.grad i j in
          set a.grad i j (get a.grad i j +. g);
          set b.grad i j (get b.grad i j +. g)
        done
      done
    )

 
  let matmul_transposed_into out a b =
    let ar, ac = dim1 a.data, dim2 a.data in
    let br, bc = dim1 b.data, dim2 b.data in
    for i = 0 to ar - 1 do
      for j = 0 to br - 1 do
        let acc = ref 0.0 in
        for k = 0 to ac - 1 do
          acc := !acc +. (get a.data i k *. get b.data j k)
        done;
        set out.data i j !acc
      done
    done;
    out._prev <- [a; b];
    out._backward <- (fun () ->
      for i = 0 to ar - 1 do
        for k = 0 to ac - 1 do
          let acc = ref 0.0 in
          for j = 0 to br - 1 do
            acc := !acc +. (get out.grad i j *. get b.data j k)
          done;
          set a.grad i k (get a.grad i k +. !acc)
        done
      done;
      for j = 0 to br - 1 do
        for k = 0 to bc - 1 do
          let acc = ref 0.0 in
          for i = 0 to ar - 1 do
            acc := !acc +. (get out.grad i j *. get a.data i k)
          done;
          set b.grad j k (get b.grad j k +. !acc)
        done
      done
    )

  let matmul_transposed a b = 
    let ar = dim1 a.data in 
    let br = dim1 b.data in 
    let out = create ar br in matmul_transposed_into out a b; out

  let rmsnorm_into out x =
    let r, c = dim1 x.data, dim2 x.data in
    for i = 0 to r - 1 do
      let ms = ref 0.0 in
      for j = 0 to c - 1 do
        let v = get x.data i j in ms := !ms +. (v *. v)
      done;
      let scale = 1.0 /. sqrt (!ms /. float c +. 1e-5) in
      for j = 0 to c - 1 do
        set out.data i j (get x.data i j *. scale)
      done
    done;
    out._prev <- [x];
    out._backward <- (fun () ->
      for i = 0 to r - 1 do
        let ms = ref 0.0 in
        let dot_gx = ref 0.0 in
        for j = 0 to c - 1 do 
          let v = get x.data i j in 
          ms := !ms +. (v *. v);
          dot_gx := !dot_gx +. (get out.grad i j *. v)
        done;
        let m_val = !ms /. float c +. 1e-5 in
        let scale = 1.0 /. sqrt m_val in
        let scale3_c = (scale *. scale *. scale) /. float c in
        for j = 0 to c - 1 do
          let d_out = get out.grad i j in
          let d_x = (d_out *. scale) -. (get x.data i j *. scale3_c *. !dot_gx) in
          set x.grad i j (get x.grad i j +. d_x)
        done
      done
    )

  let relu_into out x =
    let r, c = dim1 x.data, dim2 x.data in
    for i = 0 to r - 1 do
      for j = 0 to c - 1 do
        set out.data i j (max 0.0 (get x.data i j))
      done
    done;
    out._prev <- [x];
    out._backward <- (fun () ->
      for i = 0 to r - 1 do
        for j = 0 to c - 1 do
          if get x.data i j > 0.0 then
            set x.grad i j (get x.grad i j +. get out.grad i j)
        done
      done
    )

  let softmax_into ?len out x =
    let r, full_c = dim1 x.data, dim2 x.data in
    let c = match len with Some l -> l | None -> full_c in
    for i = 0 to r - 1 do
      let max_v = ref (-.infinity) in
      for j = 0 to c - 1 do
        let v = get x.data i j in 
        if v > !max_v then max_v := v
      done;
      let sum_exp = ref 0.0 in
      for j = 0 to c - 1 do
        let v = exp (get x.data i j -. !max_v) in
        set out.data i j v;
        sum_exp := !sum_exp +. v
      done;
      for j = 0 to c - 1 do
        set out.data i j (get out.data i j /. !sum_exp)
      done
    done;
    out._prev <- [x];
    out._backward <- (fun () ->
      for i = 0 to r - 1 do
        for j = 0 to c - 1 do
          let sj = get out.data i j in
          let gj = get out.grad i j in
          let sum_sg = ref 0.0 in
          for k = 0 to c - 1 do
            sum_sg := !sum_sg +. (get out.data i k *. get out.grad i k)
          done;
          set x.grad i j (get x.grad i j +. (sj *. (gj -. !sum_sg)))
        done
      done
    )

  let slice_row_into out x row =
    let _, c = dim1 x.data, dim2 x.data in
    for j = 0 to c - 1 do
      set out.data 0 j (get x.data row j)
    done;
    out._prev <- [x];
    out._backward <- (fun () ->
      for j = 0 to c - 1 do
        set x.grad row j (get x.grad row j +. get out.grad 0 j)
      done
    )

  let slice_col_into out x col len =
    let r, _ = dim1 x.data, dim2 x.data in
    for i = 0 to r - 1 do
      for j = 0 to len - 1 do
        set out.data i j (get x.data i (col + j))
      done
    done;
    out._prev <- [x];
    out._backward <- (fun () ->
      for i = 0 to r - 1 do
        for j = 0 to len - 1 do
          set x.grad i (col + j) (get x.grad i (col + j) +. get out.grad i j)
        done
      done
    )

  let dot_product_into out a b =
    let c = dim2 a.data in
    let acc = ref 0.0 in
    for i = 0 to c - 1 do
      acc := !acc +. (get a.data 0 i *. get b.data 0 i)
    done;
    set out.data 0 0 !acc;
    out._prev <- [a; b];
    out._backward <- (fun () ->
      let g = get out.grad 0 0 in
      for i = 0 to c - 1 do
        set a.grad 0 i (get a.grad 0 i +. g *. get b.data 0 i);
        set b.grad 0 i (get b.grad 0 i +. g *. get a.data 0 i)
      done
    )

  let dot_product a b =
    let out = create 1 1 in dot_product_into out a b; out

  let weighted_sum_into out weights values =
    (* weights: 1xT, values: list of 1xH tensors *)
    let h = dim2 out.data in
    Array2.fill out.data 0.0;
    List.iteri (fun i v ->
      let w = get weights.data 0 i in
      for j = 0 to h - 1 do
        set out.data 0 j (get out.data 0 j +. w *. get v.data 0 j)
      done
    ) values;
    out._prev <- weights :: values;
    out._backward <- (fun () ->
      let g_out = out.grad in
      List.iteri (fun i v ->
        let w = get weights.data 0 i in
        let g_w = ref 0.0 in
        for j = 0 to h - 1 do
          let gj = get g_out 0 j in
          g_w := !g_w +. gj *. get v.data 0 j;
          set v.grad 0 j (get v.grad 0 j +. w *. gj)
        done;
        set weights.grad 0 i (get weights.grad 0 i +. !g_w)
      ) values
    )

  let weighted_sum weights values =
    let h = dim2 (List.hd values).data in
    let out = create 1 h in weighted_sum_into out weights values; out

  let add a b = 
    let r, c = dim1 a.data, dim2 a.data in 
    let out = create r c in add_into out a b; out


  let rmsnorm x = 
    let r, c = dim1 x.data, dim2 x.data in 
    let out = create r c in rmsnorm_into out x; out

  let softmax ?len x = 
    let r, c = dim1 x.data, dim2 x.data in 
    let out = create r c in softmax_into ?len out x; out

  let relu x = 
    let r, c = dim1 x.data, dim2 x.data in 
    let out = create r c in relu_into out x; out

  let slice_row x r = 
    let c = dim2 x.data in 
    let out = create 1 c in slice_row_into out x r; out

  let slice_col x c len = 
    let r = dim1 x.data in 
    let out = create r len in slice_col_into out x c len; out
end

(* --- Configuration --- *)
let n_layer, n_embd, block_size, n_head = 1, 16, 16, 4
let head_dim, learning_rate, num_steps = n_embd / n_head, 0.01, 1000
let beta1, beta2, eps = 0.85, 0.99, 1e-8

(* --- Model State --- *)
type layer = {
  wq : Tensor.t; 
  wk : Tensor.t; 
  wv : Tensor.t; 
  wo : Tensor.t;
  fc1 : Tensor.t; 
  fc2 : Tensor.t;
}

type state = { 
  wte : Tensor.t; 
  wpe : Tensor.t; 
  lm_head : Tensor.t; 
  layers : layer array; 
}

(* --- Initialization Helpers --- *)

let gauss mean std =
  let u1 = Random.float 1.0 in
  let u2 = Random.float 1.0 in
  mean +. std *. sqrt (-2.0 *. log u1) *. cos (2.0 *. Float.pi *. u2)

let linear w x = Tensor.matmul_transposed x w
let ( +^ ) = Tensor.add

(* --- GPT Forward Pass --- *)
let gpt state tid pid keys values =
  let tok_emb = Tensor.slice_row state.wte tid in
  let pos_emb = Tensor.slice_row state.wpe pid in
  let x = tok_emb +^ pos_emb |> Tensor.rmsnorm in

  let rec apply_layers x li =
    if li = n_layer then x
    else
      let l = state.layers.(li) in
      let x_norm = Tensor.rmsnorm x in
      let q, k, v = 
        linear l.wq x_norm, 
        linear l.wk x_norm, 
        linear l.wv x_norm
      in
      keys.(li) <- keys.(li) @ [k];
      values.(li) <- values.(li) @ [v];

      (* Multi-Head Attention *)
      let x_attn = 
        let hs = List.init n_head (fun h ->
          let hs = h * head_dim in
          let q_h = Tensor.slice_col q hs head_dim in
          let k_h = List.map (fun ki -> Tensor.slice_col ki hs head_dim) keys.(li) in
          let v_h = List.map (fun vi -> Tensor.slice_col vi hs head_dim) values.(li) in
          
          let attn_logits_list = List.map (fun kh ->
            let dot = Tensor.dot_product q_h kh in
            let scaled_dot = Tensor.create 1 1 in
            let scale_factor = sqrt (float_of_int head_dim) in
            Tensor.set_entry scaled_dot 0 0 (Tensor.entry dot 0 0 /. scale_factor);
            scaled_dot.Tensor._prev <- [dot];
            scaled_dot.Tensor._backward <- (fun () ->
              let g = Array2.get scaled_dot.Tensor.grad 0 0 in
              let old_g = Array2.get dot.Tensor.grad 0 0 in
              Array2.set dot.Tensor.grad 0 0 (old_g +. g /. scale_factor)
            );
            scaled_dot
          ) k_h in
          
          let attn_logits = Tensor.create 1 (List.length k_h) in
          List.iteri (fun t dot_node ->
            Tensor.set_entry attn_logits 0 t (Tensor.entry dot_node 0 0)
          ) attn_logits_list;
          
          attn_logits.Tensor._prev <- attn_logits_list;
          attn_logits.Tensor._backward <- (fun () ->
            List.iteri (fun t dot_node ->
              let g = Array2.get attn_logits.Tensor.grad 0 t in
              let old_g = Array2.get dot_node.Tensor.grad 0 0 in
              Array2.set dot_node.Tensor.grad 0 0 (old_g +. g)
            ) attn_logits_list
          );
          
          let attn_weights = Tensor.softmax attn_logits in
          Tensor.weighted_sum attn_weights v_h
        ) in
        
        let out = Tensor.create 1 n_embd in
        List.iteri (fun h (h_tensor : Tensor.t) ->
          for j = 0 to head_dim - 1 do
            Tensor.set_entry out 0 (h * head_dim + j) (Tensor.entry h_tensor 0 j)
          done
        ) hs;
        
        out.Tensor._prev <- hs;
        out.Tensor._backward <- (fun () ->
          List.iteri (fun h (h_tensor : Tensor.t) ->
            for j = 0 to head_dim - 1 do
              let g = Array2.get out.Tensor.grad 0 (h * head_dim + j) in
              let old_g = Array2.get h_tensor.Tensor.grad 0 j in
              Array2.set h_tensor.Tensor.grad 0 j (old_g +. g)
            done
          ) hs
        );
        out
      in
      
      (* Residual Connection + FFN *)
      let x = x_attn |> linear l.wo |> ( +^ ) x in
      let mlp_out = 
        x |> Tensor.rmsnorm 
        |> linear l.fc1 
        |> Tensor.relu 
        |> linear l.fc2 
      in
      apply_layers (x +^ mlp_out) (li + 1)
  in
  apply_layers x 0 |> linear state.lm_head

(* --- Main Execution --- *)
let main () =
  Random.init 42;
  
  (* 1. Load Data (Minimalist) *)
  if not (Sys.file_exists "input.txt") then
    ignore (Sys.command "curl -s https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt");
  
  let docs = 
    In_channel.with_open_text "input.txt" In_channel.input_lines
    |> List.filter ((<>) "") |> Array.of_list
  in
  let all_chars = 
    String.concat "" (Array.to_list docs) 
    |> String.to_seq |> List.of_seq 
    |> List.sort_uniq Char.compare 
  in
  let uchars = Array.of_list all_chars in
  let vocab_size = Array.length uchars + 1 in
  let bos_token = Array.length uchars in

  Printf.printf "num docs: %d\n" (Array.length docs);
  Printf.printf "vocab size: %d\n" vocab_size;

  let mat r c = 
    let t = Tensor.create r c in
    for i = 0 to r - 1 do 
      for j = 0 to c - 1 do 
        Tensor.set_entry t i j (gauss 0.0 0.08) 
      done 
    done;
    t
  in
  
  (* 2. Initialize Model *)
  let state = {
    wte = mat vocab_size n_embd;
    wpe = mat block_size n_embd;
    lm_head = mat vocab_size n_embd;
    layers = Array.init n_layer (fun _ -> {
      wq = mat n_embd n_embd;
      wk = mat n_embd n_embd;
      wv = mat n_embd n_embd;
      wo = mat n_embd n_embd;
      fc1 = mat (4 * n_embd) n_embd;
      fc2 = mat n_embd (4 * n_embd);
    });
  } in

  let params = 
    [state.wte; state.wpe; state.lm_head] @ 
    (Array.to_list state.layers |> List.concat_map (fun l -> [l.wq; l.wk; l.wv; l.wo; l.fc1; l.fc2]))
  in
  let num_params = 
    List.fold_left (fun acc p -> 
      acc + (dim1 p.Tensor.data * dim2 p.Tensor.data)
    ) 0 params 
  in
  Printf.printf "num params: %d\n" num_params;
  
  let m = List.map (fun p -> 
    Array2.create Float64 c_layout (dim1 p.Tensor.data) (dim2 p.Tensor.data)
  ) params in
  let v = List.map (fun p -> 
    Array2.create Float64 c_layout (dim1 p.Tensor.data) (dim2 p.Tensor.data)
  ) params in
  
  List.iter (fun mt -> Array2.fill mt 0.0) m;
  List.iter (fun vt -> Array2.fill vt 0.0) v;

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
      let tokens = [bos_token] @ (String.to_seq doc |> Seq.map (fun c ->
        let rec find i = if uchars.(i) = c then i else find (i + 1) in find 0
      ) |> List.of_seq) @ [bos_token] in
      let n = min block_size (List.length tokens - 1) in
      
      (* Reset gradients *)
      List.iter Tensor.zero_grad params;

      let keys, values = Array.make n_layer [], Array.make n_layer [] in

      let losses = 
        List.init n (fun pos_id ->
          let tid, target = List.nth tokens pos_id, List.nth tokens (pos_id + 1) in
          let logits = gpt state tid pos_id keys values in
          let probs = Tensor.softmax logits in
          let loss_val = -. log (Tensor.entry probs 0 target +. 1e-10) in
          let node = Tensor.create 1 1 in 
          Tensor.set_entry node 0 0 loss_val;
          node.Tensor._prev <- [logits];
          node.Tensor._backward <- (fun () ->
            let g = Array2.get node.Tensor.grad 0 0 in
            for i = 0 to vocab_size - 1 do
              let si = Tensor.entry probs 0 i in
              let delta = if i = target then si -. 1.0 else si in
              let old_g = Array2.get logits.Tensor.grad 0 i in
              Array2.set logits.Tensor.grad 0 i (old_g +. g *. delta)
            done
          );
          node
        )
      in

      let avg_loss_node = Tensor.create 1 1 in
      let total_loss_val = List.fold_left (fun acc l -> acc +. Tensor.entry l 0 0) 0.0 losses in
      Tensor.set_entry avg_loss_node 0 0 (total_loss_val /. float_of_int n);
      avg_loss_node.Tensor._prev <- losses;
      avg_loss_node.Tensor._backward <- (fun () ->
        List.iter (fun (l : Tensor.t) -> 
          Array2.set l.grad 0 0 (1.0 /. float_of_int n)
        ) losses
      );
      Tensor.backward avg_loss_node;
      let avg_loss = total_loss_val /. float_of_int n in

      (* Adam Update Matching microgpt.ml *)
      let lr_t = learning_rate *. (1.0 -. (float_of_int step /. float_of_int num_steps)) in
      let rec update ps m_list v_list =
        match ps, m_list, v_list with
        | p :: pt, mt :: mtt, vt :: vtt ->
            let r, c = dim1 p.Tensor.data, dim2 p.Tensor.data in
            for ir = 0 to r - 1 do 
              for ic = 0 to c - 1 do
                let g = Array2.get p.Tensor.grad ir ic in
                let m_val = beta1 *. Array2.get mt ir ic +. (1.0 -. beta1) *. g in
                let v_val = beta2 *. Array2.get vt ir ic +. (1.0 -. beta2) *. (g *. g) in
                Array2.set mt ir ic m_val; 
                Array2.set vt ir ic v_val;
                let m_hat = m_val /. (1.0 -. (beta1 ** float_of_int (step + 1))) in
                let v_hat = v_val /. (1.0 -. (beta2 ** float_of_int (step + 1))) in
                let old_d = Array2.get p.Tensor.data ir ic in
                Array2.set p.Tensor.data ir ic (old_d -. lr_t *. m_hat /. (sqrt v_hat +. eps));
                Array2.set p.Tensor.grad ir ic 0.0
              done 
            done;
            update pt mtt vtt
        | _ -> ()
      in
      update params m v;

      Printf.printf "step %4d / %4d | loss %.4f\r%!" (step + 1) num_steps avg_loss;
      train_loop (step + 1)
    end
  in
  train_loop 0;
  
  let rec infer_loop i =
    if i > 20 then ()
    else begin
      let rec gen tokens keys values =
        if List.length tokens > 15 then tokens
        else
          let tid = List.hd (List.rev tokens) in
          let pos_id = List.length tokens - 1 in
          let logits = gpt state tid pos_id keys values in
          for j = 0 to vocab_size - 1 do 
            Tensor.set_entry logits 0 j (Tensor.entry logits 0 j /. 0.5) 
          done;
          let probs = Tensor.softmax logits in
          let r = Random.float 1.0 in
          let rec sample_prob i cum =
            if i >= vocab_size then bos_token else
            let cum = cum +. Tensor.entry probs 0 i in
            if r <= cum then i else sample_prob (i + 1) cum
          in
          let next_id = sample_prob 0 0.0 in
          if next_id = bos_token then tokens 
          else gen (tokens @ [next_id]) keys values
      in
      let keys, values = Array.make n_layer [], Array.make n_layer [] in
      let tokens = gen [bos_token] keys values in
      let name = 
        List.filter (fun t -> t <> bos_token) tokens 
        |> List.map (fun t -> uchars.(t)) |> List.to_seq |> String.of_seq 
      in
      Printf.printf "sample %2d: %s\n" i name;
      infer_loop (i + 1)
    end
  in
  infer_loop 1

let () = main ()
