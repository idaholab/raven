global_param_1 = 0.1
global_param_2 = 0.2

[ParserChallenges]
  scalar = 3.14 # scalarComment
  scalar2 =

       6.28

  scalar3 = ${global_param_1}
  scalar4 = ${global_param_2}

  # scalarComment
  string = OneString # stringComment

  # linebreak comment

  onelineVector = 'A B C D E'
  onelineComment = 'F G H I J' # commentOnOneLineVector
  multilineVector = 'A1 B1 C1 D1
                     A2 B2 C2 D2
                     A3 B3 C3 D3'
  multilineVectorHanging = 'A1 B1 C1 D1
                            A2 B2 C2 D2
                            A3 B3 C3 D3
'
  multilineVectorNewlines =
'
  A1 B1 C1 D1
  A2 B2
  A3 B3 C3 D3
'
  multilineHangingComment = '
  A1 B1 C1 D1
  A2 B2 C2 D2
  A3 B3 C3 D3
' # commentMultilineHanging

  [./Nested]
    nested_scalar = 3.14 # scalarComment
    nested_scalar2 =

        6.28


    nestedVector = 'A B C D E'
    nestedMultiline =
'
A1 B1 C1 D1
A2 B2
A3 B3 C3 D3
'
    [./NestedTwo]
      nested_two_scalar = 3.14 # scalarComment
      nested_two_vector = 'A B C D E'
      nested_two_Multiline =
        '
        A1 B1 C1 D1
        A2 B2
        A3 B3 C3 D3
        '
    [../]
  [../]
[]

[Outputs]
  [./csv]
    type = CSV
  [../]
[]
