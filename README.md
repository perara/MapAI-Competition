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

### Steps

#### Step 1 - Fork

Fork the [MapAI-Competition](https://github.com/Sjyhne/MapAI-Competition) repository in Github.
Forking creates a clone of the base repo on your own user and allows for easier pull requests
and so on.

#### Step 2 - Clone with -o parameter

Clone your fork down to your computer with the following command:

`git clone git@github.com:<your_username>/MapAI-Competition.git -o submission`

The _-o_ parameter sets the origin name for this repostory to be "_submission_" and not the
default which is "_origin_".

#### Step 3 - Create a new private (or public) repository

Create a new private repository on your own github. The reason we need this is because it is
not possible to set the visibility of a fork to private. Therefore, to keep your development progress
private, we have to add another remote repository for the MapAI-Competition fork.

To do this, you have to change directories into the cloned fork. E.g. `cd MapAI-Competition`.

#### Step 4 - Add private remote repository to fork

Then, we can keep developing in the cloned fork and push the changes to the private repository.
To be able to do this, we have to add remote origin by running the following command:

`git remote add origin <private_repository>`

E.g.

`git remote add origin git@github.com:Sjyhne/my_private_repository.git`

This will enable you to push your changes to the private repository and not the public fork
by just pushing as usual to origin master. Because we have not specified the origin for the remote 
it will default to _origin_.

`git push origin <branch>`

#### Step 5 - Create your own team-folder

It is important to follow the structure of the team_template in the repository. The easiest way to
keep this structure is by just creating a copy of the team_template folder and name it according
to your team name. The folder you create must follow the correct naming structure, which is 
`team_<team_name>`. You can copy the team_template and name it with the following command:

`cp -r team_template ./team_<team_name>`

For the entirety of the competition, you will only change and develop inside this folder. Nothing
outside the team-folder should be changed or altered. You can find more information about
the folder structure and contents in the section about _folder structure_.

The template is already ready with code that can run, train, and evaluate - this is just template
code and you are allowed to change everything related to the training of the models. When it comes
the evaluation files, it is more restricted, as they are used to automatically evaluate the models.



#### Step 6 - Delivery

When the deadline are due, there are a few steps that will have to be taken to get ready for
submission.

##### Step 6.1 - Push your changes to the fork

Push all of your changes to the fork - this will make your code and models visible in the fork.
This is done by running the following command:

`git push submission master`

As we set the origin for the fork to _submission_ in the start.

##### 6.2 - Create a pull request to the base repo

The next step is to create a pull request against the base repository. This will initiate a 
workflow that runs and evaluates the model on a validation dataset. This workflow will have to
pass in order to deliver the submission.


When the deadline is finished, we will evaluate all of your code on the hidden test-dataset and publish the results
on a github page.

NB: It is important that the submission is formatted correctly and is the correct resolution.

## Downloading data

The training and validation data is stored in separate github repositories.
Therefore, all you need to do is clone both repositories into a data folder
located at the same level as your team folder. Further instructions can be
found in the [readme](./data/README.md) in the data folder.

## Evaluation

We will evaluate each of the tasks in each submission using Github Actions, which requires that the submissions
are formatted correctly and outputs files with the correct name, type, and resolution.

## Example

We provide you with an example project called team_template. The team_template contains example code for training and submission
and show you what we expect the outcome to be. 

We will also provide you with tests that will check certain conditions about your current
submission format and state. The test will use a small test-set to verify the correctness of the delivery.
