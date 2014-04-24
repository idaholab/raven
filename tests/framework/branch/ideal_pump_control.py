import distribution1D
distcont  = distribution1D.DistributionContainer.Instance()


def initial_function(monitored, controlled, auxiliary):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    auxiliary.dummy_for_branch = 0.0

    return

def control_function(monitored, controlled, auxiliary):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    mult = 1.01
    if auxiliary.dummy_for_branch < 1.0:
        auxiliary.dummy_for_branch = auxiliary.dummy_for_branch + 0.25
    print('THRESHOLDDDDDD ' + str(distributions.zeroToOne.getVariable('ProbabilityThreshold')))
    return

def dynamic_event_tree(monitored, controlled, auxiliary):
  print("######################################## In dynamic_event_tree")
  if distcont.checkCdf('zeroToOne',auxiliary.dummy_for_branch) and (not auxiliary.aBoolean) and monitored.time_step>1:
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    auxiliary.aBoolean = True
  return
