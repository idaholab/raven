[./Simulation]
  verbosity = debug

  [./RunInfo]
    [./WorkingDir]
      value = GridTest_GetPot
    [../]
    [./Sequence]
      value = CustomSampler
    [../]
    [./batchSize]
      value = 1
    [../]
  [../]

  [./Models]
    [./Dummy]
      name = MyDummy
      subType =
    [../]
  [../]

  [./Distributions]
    [./Normal]
      name = Gauss1
      [./mean]
        value = 1
      [../]
      [./sigma]
        value = 0.001
      [../]
      [./lowerBound]
        value = 0
      [../]
      [./upperBound]
        value = 2
      [../]
    [../]
    [./Normal]
      name = Gauss2
      [./mean]
        value = 1
      [../]
      [./sigma]
        value = 0.4
      [../]
      [./lowerBound]
        value = 0
      [../]
      [./upperBound]
        value = 2
      [../]
    [../]
    [./Triangular]
      name = DistTri1
      [./apex]
        value = 1
      [../]
      [./min]
        value = -0.1
      [../]
      [./max]
        value = 4
      [../]
    [../]
  [../]

  [./Samplers]
    [./Grid]
      name = myGrid
      [./variable]
        name = VarGauss1
        [./distribution]
          value = Gauss1
        [../]
        [./grid]
          construction = custom
          type = value
          value = '0.02 0.5 0.6'
        [../]
      [../]
      [./variable]
        name = VarGauss2
        [./distribution]
          value = Gauss2
        [../]
        [./grid]
          construction = custom
          type = CDF
          value = '0.5 1.0 0.0'
        [../]
      [../]
      [./variable]
        name = VarTri1
        [./distribution]
          value = DistTri1
        [../]
        [./grid]
          construction = equal
          steps = 1
          type = value
          value = '3.5 4.0'
        [../]
      [../]
    [../]
  [../]

  [./DataObjects]
    [./PointSet]
      name = outGrid
      [./Input]
        value = VarGauss1,VarGauss2,VarTri1
      [../]
      [./Output]
        value = OutputPlaceHolder
      [../]
    [../]
    [./PointSet]
      name = dummyIN
      [./Input]
        value = VarGauss1,VarGauss2,VarTri1
      [../]
      [./Output]
        value = OutputPlaceHolder
      [../]
    [../]
  [../]

  [./OutStreams]
    [./Print]
      name = outGrid_dump
      [./type]
        value = csv
      [../]
      [./source]
        value = outGrid
      [../]
    [../]
  [../]

  [./Steps]
    [./MultiRun]
      name = CustomSampler
      [./Input]
        class = DataObjects
        type = PointSet
        value = dummyIN
      [../]
      [./Model]
        class = Models
        type = Dummy
        value = MyDummy
      [../]
      [./Sampler]
        class = Samplers
        type = Grid
        value = myGrid
      [../]
      [./Output]
        class = DataObjects
        type = PointSet
        value = outGrid
      [../]
      [./Output]
        class = Databases
        type = HDF5
        value = test_DummyModel_db
      [../]
      [./Output]
        class = OutStreams
        type = Print
        value = outGrid_dump
      [../]
    [../]
  [../]

  [./Databases]
    [./HDF5]
      name = test_DummyModel_db
      readMode = overwrite
    [../]
  [../]
[../]
