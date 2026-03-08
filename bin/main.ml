open Microgpt

let () =
  if Array.length Sys.argv > 0 && (Filename.basename Sys.argv.(0) = "ocamlgpt" || Filename.basename Sys.argv.(0) = "main.exe" || Filename.basename Sys.argv.(0) = "main") then
    main ()
