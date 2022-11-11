The run_raven_test.sh and checkout_build_and_test.sh are designed to be used
for automatically building and testing raven.

It can be used from crontab or some other way of automatically running it.

For example, here is the crontab entry that is used to run it once a
weekday and email the result to the mailing list (where
$HOME/bin/run_raven_test.sh is symbolically linked to the version
inside of raven):

0       2       *       *       1-4     RESULT_EMAIL=raven-devel@inl.gov $HOME/bin/run_raven_test.sh

There are three environmental variables that can be used to customize
the script.

FROM_EMAIL specifies what the email address that the code should say
it comes from and the default is git config user.email.

RESULT_EMAIL specifies where to email the results of the test and
defaults to FROM_EMAIL

PROJECT_DIR specifies what directory to checkout and build raven into,
and defaults to $HOME/test_projects

In order for the run_raven_test.sh script to work, it needs a cluster
where
/usr/bin/modulecmd bash load pbs raven-devel-gcc
gets the environment ready to compile raven.  Alternatively, the
run_raven_test.sh can be copied and modified for the cluster it is running
on.

