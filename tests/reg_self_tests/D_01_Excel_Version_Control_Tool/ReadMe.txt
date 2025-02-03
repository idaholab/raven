	• Prerequisite: 
		○ The user would need to install raven and put the excel file (in the format of either .xlsx, .xlsb, .xla, .xlam, .xlsm, .xlt, .xltm, .xltx) that need to be version-controlled under \projects\raven\tests\reg_self_tests\D_01_Excel_Version_Control_Tool.
		○ Save and commit and changes for the tool.
	• Objectives:
		○ This tool is developed to help the users control the versions of the excel files. The differences of the existing version and the previous version would be documented so that the user can decide whether to modify the content and resume and previous version at any time.
		○ Track the file using "git add excel_file_name" to perform version control of the excel file
	• Steps for version-control
		○ You must close the excel file before executing the "git diff"
		○ There are two ways for the version control. The first is to compare the excel files in the same branch [1] while the other condition is to compare the excel files in the different branches [2]. Please follow the instructions bellow to diff the two excel files: 
			§ [1] In the same branch
				□ Make changes and save the excel book
				□ Run "git status" to make sure some changes has been made
				□ Run "git diff <PathToFile/file_name>" and wait until it finished
				□ Check the "Diff_Results.txt" in the same folder
			§ [2] In the different branches
				□ Create a new branch using "git branch new"
				□ Switch to the "new" branch from "master" branch
				□ Open an excel to make changes. Then save and close the file.
				□ Use "git add "file name"" to accept the changes of the file
				□ Commit the changes using "git commit -m "adding text over here"
				□ Use "git diff master…HEAD" (meaning to compare the file in the new branch to the file in the master branch)
				□ Option to merge to the master branch
					® Git checkout master
					® Git merge new
					® *note that after following this, the git diff will be empty.
		○ The outputs (Diff_Results.txt) of the comparisons of the "git diff" will show the difference between the two different versions of the excel if you made some changes. You will need to review the changes row by row and see if you agree. If yes, you would need to type "git add excel_file_name". Then, type "git commit -m "text to commit"". The committed text would be saved in the log file under git repository. 
If you do not agree with the changes, you would need to type "git restore excel_file_name" to remove the changes and back to the original version.