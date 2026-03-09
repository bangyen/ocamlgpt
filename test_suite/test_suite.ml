open Microgpt
open Printf

let assert_float_equal name expected actual epsilon =
  if abs_float (expected -. actual) > epsilon then begin
    printf "FAIL: %s (expected %.6f, got %.6f)\n" name expected actual;
    exit 1
  end else
    printf "PASS: %s\n" name

let test_autograd_basic () =
  printf "--- Testing Autograd Basic Operations ---\n";
  let open Value in
  let a = create (-4.0) in
  let b = create 2.0 in
  let c = a +: b in
  let d = (a *: b) +: (b *: b *: b) in
  let c = c +: (c +: create 1.0) in
  let c = c +: (create 1.0 +: c +: (create (-1.0) *: a)) in
  let d = d +: (d *: (create 2.0)) +: ((b +: a) *: (create 1.0)) in
  let d = d +: (d *: (create 3.0)) +: (create 10.0 +: (b *: b)) in
  let e = c -: d in
  let f = e *: e in
  let g = (f /: (create 2.0)) +: (create 10.0 /: f) in
  let h = log (create 2.0 +: (g *: g)) in
  
  backward h;
  
  assert_float_equal "h.data" 6.417209 h.data 1e-4; 
  printf "Final h.data: %.6f\n" h.data;
  printf "Gradients: a.grad=%.6f, b.grad=%.6f\n" a.grad b.grad;
  ()

let test_relu () =
  printf "\n--- Testing ReLU ---\n";
  let open Value in
  let x = create (-5.0) in
  let y = relu x in
  backward y;
  assert_float_equal "relu(-5) data" 0.0 y.data 1e-9;
  assert_float_equal "relu(-5) grad" 0.0 x.grad 1e-9;
  
  let x2 = create 5.0 in
  let y2 = relu x2 in
  backward y2;
  assert_float_equal "relu(5) data" 5.0 y2.data 1e-9;
  assert_float_equal "relu(5) grad" 1.0 x2.grad 1e-9;
  ()

let test_softmax () =
  printf "\n--- Testing Softmax ---\n";
  let open Value in
  let logits = [|create 1.0; create 2.0; create 0.0|] in
  let probs = softmax logits in
  let total_prob = Array.fold_left (fun acc p -> acc +. p.data) 0.0 probs in
  assert_float_equal "softmax total prob" 1.0 total_prob 1e-6;
  printf "Softmax output: [%s]\n" (String.concat ", " (Array.to_list (Array.map (fun p -> string_of_float p.data) probs)));
  ()

let test_rmsnorm () =
  printf "\n--- Testing RMSNorm ---\n";
  let open Value in
  let x = [|create 1.0; create 2.0; create 3.0|] in
  let y = rmsnorm x in
  let ms = (Array.fold_left (fun acc xi -> acc +. (xi.data *. xi.data)) 0.0 y) /. 3.0 in
  assert_float_equal "rmsnorm output ms" 1.0 ms 1e-5;
  ()

let test_gradient_check () =
  printf "\n--- Numerical Gradient Check (Finite Differences) ---\n";
  let open Value in
  let x = create 2.0 in
  let w = create (-3.0) in
  let b = create 1.0 in
  let f_loss v_w =
    let layer_out = (x *: v_w) +: b in
    layer_out *: layer_out
  in
  let loss = f_loss w in
  backward loss;
  let analytic_grad = w.grad in
  
  let eps = 1e-6 in
  let w_plus = create (w.data +. eps) in
  let loss_plus = f_loss w_plus in
  let w_minus = create (w.data -. eps) in
  let loss_minus = f_loss w_minus in
  let numerical_grad = (loss_plus.data -. loss_minus.data) /. (2.0 *. eps) in
  
  assert_float_equal "numerical vs analytic grad" numerical_grad analytic_grad 1e-5;
  printf "Analytic: %.6f, Numerical: %.6f\n" analytic_grad numerical_grad;
  ()

let test_linear_backward () =
  printf "\n--- Testing Linear Layer Backward ---\n";
  let open Value in
  let x = [|create 1.0; create (-2.0)|] in
  let w = [| [|create 0.5; create (-0.1)|]; [|create 0.3; create 0.8|] |] in
  let out = linear x w in
  let loss = out.(0) *: out.(0) +: out.(1) *: out.(1) in
  backward loss;
  (* If loss = sum(y_i^2), then dL/dy_i = 2*y_i.
     y_0 = x_0*w_00 + x_1*w_01 = 1*0.5 + (-2)*(-0.1) = 0.7
     y_1 = x_0*w_10 + x_1*w_11 = 1*0.3 + (-2)*0.8 = -1.3
     dL/dw_00 = dL/dy_0 * dy_0/dw_00 = 2*y_0 * x_0 = 1.4 * 1 = 1.4
  *)
  assert_float_equal "dL/dw_00" 1.4 w.(0).(0).grad 1e-6;
  assert_float_equal "dL/dw_11" (2.0 *. (-1.3) *. (-2.0)) w.(1).(1).grad 1e-6;
  ()

let test_optimizer_update () =
    printf "\n--- Testing Weight Updates ---\n";
    let open Value in
    let param = create 10.0 in
    param.grad <- 1.0;
    (* Simple SGD step emulation to verify mutability *)
    let lr = 0.1 in
    param.data <- param.data -. lr *. param.grad;
    assert_float_equal "param update" 9.9 param.data 1e-9;
    ()

let () =
  test_autograd_basic ();
  test_relu ();
  test_softmax ();
  test_rmsnorm ();
  test_gradient_check ();
  test_linear_backward ();
  test_optimizer_update ();
  printf "\nAll strong tests passed!\n"
