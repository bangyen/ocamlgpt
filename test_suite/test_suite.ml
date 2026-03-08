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
  let logits = [create 1.0; create 2.0; create 0.0] in
  let probs = softmax logits in
  let total_prob = List.fold_left (fun acc p -> acc +. p.data) 0.0 probs in
  assert_float_equal "softmax total prob" 1.0 total_prob 1e-6;
  printf "Softmax output: [%s]\n" (String.concat ", " (List.map (fun p -> string_of_float p.data) probs));
  ()

let test_rmsnorm () =
  printf "\n--- Testing RMSNorm ---\n";
  let open Value in
  let x = [create 1.0; create 2.0; create 3.0] in
  let y = rmsnorm x in
  let ms = (List.fold_left (fun acc xi -> acc +. (xi.data *. xi.data)) 0.0 y) /. 3.0 in
  assert_float_equal "rmsnorm output ms" 1.0 ms 1e-5;
  ()

let () =
  test_autograd_basic ();
  test_relu ();
  test_softmax ();
  test_rmsnorm ();
  printf "\nAll tests passed!\n"
