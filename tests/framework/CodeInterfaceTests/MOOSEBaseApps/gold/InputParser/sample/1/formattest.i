[ParserChallenges]
  scalar = -0.250919771206
  scalar2 = 6.28
  string = OneString
  onelineVector = "A B C D E"
  onelineComment = "F G H I J"
  multilineVector = "A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3"
  multilineVectorHanging = "A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3"
  multilineVectorNewlines = "A1 B1 C1 D1 A2 B2 A3 B3 C3 D3"
  multilineHangingComment = "A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3"
  [./Nested]
    nested_scalar = 3.14
    nested_scalar2 = 6.28
    nestedVector = "A B C D E"
    nestedMultiline = "A1 B1 C1 D1 A2 B2 A3 B3 C3 D3"
    [./NestedTwo]
      nested_two_scalar = 3.14
      nested_two_vector = "A B C D E"
      nested_two_Multiline = "A1 B1 C1 D1 A2 B2 A3 B3 C3 D3"
    [../]
  [../]
[]
[Outputs]
  csv = true
  file_base = out~formattest
[]
