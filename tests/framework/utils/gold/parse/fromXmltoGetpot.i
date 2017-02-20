[./MyRoot]
  ratrib1 = RA1
  ratrib2 = RA2

  [./FirstBlockTag]
    fbtatrib = FBTA
    [./SubTag]
      subattrib = SA
      value = subtext
    [../]
  [../]

  [./DeepOne] #comment about DeepOne
    do = D1
    [./DeepTwo] #comment about DeepTwo
      do = D2
      [./DeepThree]
        do = D3
        [./DeepFour]
          do = D4
          value = deepest
        [../]
      [../]
    [../]
    [./NotSoDeep]
      do = NSD
    [../]
  [../]

  [./Objects] #generic comment
    [./Object]
      name = MyObj
      type = OtherObj
      [./params]
        value = 1,2,3
      [../]
      [./alpha]
        value = a,b,c
      [../]
    [../]
    [./Object]
      name = MyObj2
      type = OtherObj
      [./params]
        value = 7,8,9
      [../]
      [./alpha]
        value = x,y,z
      [../]
    [../]
  [../]
[../]
