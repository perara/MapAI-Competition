# MapAI Competition

This is the official repository for the MapAI competition arranged by 
The Norwegian Mapping Authority, University of Agder (UiA),
Norwegian Artificial Intelligence Research Consortium (NORA), 
AI:Hub, and Norkart.

For this competition we are testing a new competition-framework/template for arranging 
competitions using Github and Huggingface. This is the first competition we are arranging with
the framework, and therefore appreciate your feedback and will be available for questions.
If there is a question or something is wrong, please raise a issue regarding the question.

## Competition Information

The competition aims at building segmentation using aerial images and lidar data. The 
competition consists of two different tasks:

1. Building segmentation only using aerial images.
2. Building segmentation using lidar data (it is allowed to combine the lidar with aerial images).

### Dataset

The training and validation data is an open building segmentation dataset from Denmark. While the
test data comes from various locations in Norway which is currently not released yet. All
data will be released after the competition in a collected huggingface dataset.

It is important to note that the training and evaluation data comes from real-world data. As a
result there will be flaws in the dataset, such as missing masks or masks that does not correspond
to a building in an image.

The images come from orthophotos generated using a DTM, and therefore, the building-masks are
slightly skewed compared to the ground truth masks.

### Motivation

Acquiring accurate segmentation masks of buildings is challenging since the training data 
derives from real-world photographs. As a result, the data often have varying quality, 
large class imbalance, and contains noise in different forms. The segmentation masks 
are affected by optical issues such as shadows, reflections, and perspectives. Additionally, 
trees, powerlines, or even other buildings may obstruct the visibility. 
Furthermore, small buildings have proved to be more difficult to segment than larger ones as 
they are harder to detect, more prone to being obstructed, and often confused with other classes. 
Lastly, different buildings are found in several diverse areas, ranging 
from rural to urban locations. The diversity poses a vital requirement for the model to 
generalize to the various combinations. These hardships motivate the competition and our 
evaluation method.

### Rules

Below is a set of rules that is necessary to follow in order to not be disqualified

* You are not allowed to alter the training and validation splits - they must stay as is.
* You are not allowed to use other training data than what is provided through the project 
(However, pretrained models are allowed).
* You are encouraged to create teams, but submissions must come from the same fork.
* You are not allowed to alter the evaluation.py file.
* evaluate_task_*.py can only be altered in such a fashion that it is able to run your model.

### Registration

Forking the base repository is part of the registration process. However, we would also like you to register your information
in a Google forms. The main reason comes from the ability to reach out to the participant with important information. 
As mentioned earlier, the competition framework is piloting with this competition, and therefore it will be more
prone to change compared to more established competition frameworks.

[Google form]()

## Instructions

The competition will be arranged on Github. The steps for participation is as following:

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
`team_<team_name>` (Please make it kinda unique, e.g. two first letters of each teammate). 
You can copy the team_template and name it with the following command:

`cp -r team_template ./team_<team_name>`

For the entirety of the competition, you will only change and develop inside this folder. Nothing
outside the team-folder should be changed or altered. You can find more information about
the folder structure and contents in the section about _folder structure_.

The template is already ready with code that can run, train, and evaluate - this is just template
code and you are allowed to change everything related to the training of the models. When it comes
the evaluation files, it is more restricted, as they are used to automatically evaluate the models.



#### Step 6 - Delivery

When the deadline is due, there are a few steps that will have to be taken to get ready for
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


### Uploading and downloading models from Huggingface

As the .pt files for models can be rather large, we want you to upload the model files to
your own google drive and download them from there during evaluation.

1. Train a model and get the .pt file
2. Upload the modelfile to a google drive folder
3. Enable sharing with link (Which allows us to use gdown to download the modelfile)
4. Get the link
   * Most likely looking like: "https://drive.google.com/file/d/1wFHRUDe29a82fof1LNwFsFpOvX0StWL5/view?usp=sharing"
   * Then just put the sharing link into the _pt_share_link_ variable in both evaluate_task_*.py files which will
     get the id and put it into the correct format
5. Test the modelfiles and check that it is correct and loads correctly
6. During submission, ensure this works correctly by following the "Checklist before
    submission" section

### Checklist before submission

* Verify that, from inside the src folder, you can run the following commands:
  * `python3 evaluate_task_1.py --dtype validation --device cuda:0/cpu`
  * `python3 evaluate_task_2.py --dtype validation --device cuda:0/cpu`
  * `python3 evaluation.py --task 1 --dtype validation --device cuda:0/cpu`
  * `python3 evaluation.py --task 2 --dtype validation --device cuda:0/cpu`
* Verify that you are able to run the evaluation pipeline
  * To do this you can go to your own repository.
  * Then go to the actions tab
  * Press the "Evaluation Pipeline"
  * Then press the "run workflow" for the branch you are testing (most likely master)
  * Ensure the entire workflow runs without any issues (environment issues are common)
    * If the environment issues are an issue, then you have to edit the pyproject.toml in
      the base of your team folder