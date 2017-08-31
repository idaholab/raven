----------------
For Change Control Board: Change Request Review
----------------
The following review must be completed by an authorized member of the Change Control Board.
- [ ] 1. Review all computer code.
- [ ] 2. If any changes occur to the input syntax, there must be an accompanying change to the user manual and xsd schema. If the input syntax change deprecates existing input files, a conversion script needs to be added (see Conversion Scripts).
- [ ] 3. Make sure the Python code and commenting standards are respected (camelBack, etc.) - See on the [wiki](https://github.com/idaholab/raven/wiki/RAVEN-Code-Standards#python) for details.
- [ ] 4. Automated Tests should pass, including run_tests, pylint, manual building and xsd tests. If there are changes to Simulation.py or JobHandler.py the qsub tests must pass.
- [ ] 5. If significant functionality is added, there must  be tests added to check this. Tests should cover all possible options.  Multiple short tests are preferred over one large test. If new development on the internal JobHandler parallel system is performed, a cluster test must be added setting, in <RunInfo> XML block, the node ```<internalParallel>``` to True.
- [ ] 6. If the change modifies or adds a requirement or a requirement based test case, the Change Control Board's Chair or designee also needs to approve the change.  The requirements and the requirements test shall be in sync.
- [ ] 7. The merge request must reference an issue.  If the issue is closed, the issue close checklist shall be done.
- [ ] 8. If an analytic test is changed/added is the the analytic documentation updated/added?


----------------
For Change Control Board: Issue Review
----------------
This review should occur before any development is performed as a response to this issue.
- [ ] 1. Is it tagged with a type: defect or improvement?
- [ ] 2. Is it tagged with a priority: critical, normal or minor?
- [ ] 3. If it will impact requirements or requirements tests, is it tagged with requirements?
- [ ] 4. If it is a defect, can it cause wrong results for users? If so an email needs to be sent to the users.
- [ ] 5. Is a rationale provided? (Such as explaining why the improvement is needed or why current code is wrong.)

-------
For Change Control Board: Issue Closure
-------
This review should occur when the issue is imminently going to be closed.
- [ ] 1. If the issue is a defect, is the defect fixed?
- [ ] 2. If the issue is a defect, is the defect tested for in the regression test system?  (If not explain why not.)
- [ ] 3. If the issue can impact users, has an email to the users group been written (the email should specify if the defect impacts stable or master)?
- [ ] 4. If the issue is a defect, does it impact the latest stable branch? If yes, is there any issue tagged with stable (create if needed)?
- [ ] 5. If the issue is being closed without a merge request, has an explanation of why it is being closed been provided?
