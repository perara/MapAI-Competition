# MapAI Competition

This is the official repository for the MapAI competition arranged by The Norwegian Mapping Authority, University of Agder (UiA),
Norwegian Artificial Intelligence Research Consortium (NORA), Mechatronics Innovation Lab (MIL), and Norkart.

## Instructions

The competition will be arranged on Github. The steps for participation is as following:

1. Fork this repository
2. `git clone git@github.com:perara/MapAI-Competition.git -o submission `
3. Create new repository on Github. This repository can be private.
4. Enter the MapAI-Competition folder
5. `git remote add origin git@github.com:uiaikt/map-ai-submission-test.git`
6. Create Model for the competition and use git as you would normally.
7. Before deadline:
   * `git push submission`
   * Create pull request

When the deadline is finished, we will evaluate all of your code on the hidden test-dataset and publish the results
on a github page.

NB: It is important that the submission is formatted correctly and is the correct resolution.

## Evaluation

We will evaluate each of the tasks in each submission using Github Actions, which requires that the submissions
are formatted correctly and outputs files with the correct name, type, and resolution.

## Example

We provide you with an example project called team_template. The team_template contains example code for training and submission
and show you what we expect the outcome to be. 

We will also provide you with tests that will check certain conditions about your current
submission format and state. The test will use a small test-set to verify the correctness of the delivery.
