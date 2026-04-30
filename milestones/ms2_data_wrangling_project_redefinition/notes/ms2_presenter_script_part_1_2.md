# Intro and Data Script

## Slide 1. Project Motivation

Approx. time: 45 to 60 seconds

“Our project asks whether emotion-related language can help explain self-reported MBTI patterns in Reddit writing. We think this is interesting for two reasons. First, emotion is relatively interpretable, so it can give us more insight than a purely black-box text model. Second, MBTI labels are noisy and self-reported, so before we build models we need to understand what the data can actually support. For milestone 2, our goal was to validate the data pipeline, inspect the data quality, and figure out whether the original modeling plan was realistic.”

## Slide 2. Data Acquisition and Preparation

Approx. time: 60 to 75 seconds

“We used two public datasets. The first is a balanced emotion dataset from Hugging Face with 20,000 short texts and six emotion labels. It is small, clean, and useful as a source task for learning emotion. The second is a much larger Kaggle dataset with about 13 million Reddit posts from 11,773 authors, each associated with a self-reported MBTI type. In the notebook, we standardized the schema by renaming columns, upper-casing the MBTI labels, and removing 180 blank Reddit posts. We also derived the four binary MBTI dimensions: E over I, N over S, F over T, and J over P. Finally, because the Reddit file is 2.7 gigabytes, we used the full dataset for count-based summaries and a fixed 250-thousand-post sample for more expensive text-level EDA.”

## Slide 3. EDA: Clean Source, Difficult Target

Approx. time: 60 to 75 seconds

“This slide shows the key contrast between the two datasets. On the left, the emotion dataset is almost perfectly balanced across the six classes, so it is a strong and well-behaved source dataset. On the right, the Reddit MBTI labels are extremely imbalanced. INFP alone makes up about 23 percent of all posts, while ESFP is less than two-tenths of one percent. So the first major insight from EDA is that the source task is clean, but the target task is much harder. That means a naive 16-class MBTI classifier would be a weak starting point, and it pushes us toward a more careful problem definition.”

## Slide 4. EDA: Author Concentration Changes the Modeling Plan

Approx. time: 60 seconds

“The second major insight is that the Reddit dataset is not a simple table of independent examples. The median author contributes 272 posts, the 99th percentile contributes more than 12,000, and the top 1 percent of authors generate almost 19 percent of the corpus. That means a random post-level split would leak author-specific writing style across train and test. So the main modeling implication is that we need author-level splitting and evaluation. This also motivates the transition to the next section of the talk, where we redefine the scope toward the four binary MBTI dimensions rather than a full 16-way prediction problem.”

## Delivery Notes

- Keep your section to about 3.5 to 4 minutes.
- Spend the most time on slides 3 and 4, because those are the slides with the strongest evidence.
- End slide 4 by explicitly handing off: “Based on these data constraints, we rescaled the problem as follows.”


## Slide 5. Redefinition and Rescoping of the Problem Statement

Approx. time: 45 to 60 seconds

“Based on the EDA, we changed the problem. At first, we thought about direct 16-type MBTI classification from Reddit posts. But the data does not support that plan well. The classes are very imbalanced, the labels are self-reported, and many posts come from the same authors. So instead, we focus on the four binary MBTI dimensions: E versus I, N versus S, F versus T, and J versus P. We also move from post-level prediction to user-level analysis. Our new question is: can emotion patterns in Reddit writing help explain these four MBTI dimensions?”

## Slide 6. Delineation of Next Steps by Team Members

Approx. time: 45 to 60 seconds

“Our next step is a two-stage pipeline. First, we train an emotion classifier on the Twitter emotion dataset, because that dataset has emotion labels. Second, we apply that model to Reddit posts and get predicted emotion scores. Then we group posts by user and build a user-level emotion profile. After that, we study whether these emotion features help explain the four MBTI dimensions. In terms of team work, one part is model training, one part is Reddit feature building, and one part is downstream MBTI analysis and evaluation.”

## Slide 7. Future Considerations

Approx. time: 45 to 60 seconds

“This is a deep learning project, but not every step needs a deep model. The deep learning part is the transformer-based emotion classifier. For the downstream MBTI task, we will start with simple models such as logistic regression, and then compare with a stronger baseline if needed. There are still some concerns. The MBTI labels are noisy, Reddit language may be different from the Twitter emotion data, and some users write much more than others. So one question for our TF is whether this scope is right for the final project, and whether this two-stage design is a good balance between model strength and interpretability.”