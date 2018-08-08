[./Simulation]
  verbosity = debug
  [./RunInfo]
    [./WorkingDir]
      value = ExternalXMLTest
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
      subType = ''
    [../]
  [../]

  [./ExternalXML]
    node = Distributions
    xmlToLoad = ExternalXMLTest/external_distributions.xml
  [../]
  [./ExternalXML]
    node = Samplers
    xmlToLoad = ExternalXMLTest/external_samplers.xml
  [../]

  [./DataObjects]
    [./ExternalXML]
      node = PointSet
      xmlToLoad = ExternalXMLTest/external_pointset1.xml
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
      [./ExternalXML]
        node = type
        xmlToLoad = ExternalXMLTest/external_printtype.xml
      [../]
      [./ExternalXML]
        node = source
        xmlToLoad = ExternalXMLTest/external_printsource.xml
      [../]
      [./what]
        value = input,output
      [../]
    [../]
  [../]

  [./Steps]
    [./ExternalXML]
      node = MultiRun
      xmlToLoad = ExternalXMLTest/external_multirun.xml
    [../]
  [../]

  [./Databases]
    [./HDF5]
      name = test_DummyModel_db
      readMode = overwrite
    [../]
  [../]
[../]
