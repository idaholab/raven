#!/bin/bash

for DIR in user_manual qa_docs tests; do
    (cd $DIR && make)
done

mkdir pdfs
for DOC in qa_docs/raven_sdd.pdf qa_docs/test_plan.pdf qa_docs/requirements.pdf  user_manual/raven_user_manual.pdf tests/analytic_tests.pdf; do
    cp $DOC pdfs/
done


