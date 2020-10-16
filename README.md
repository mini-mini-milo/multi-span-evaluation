# IMPORTANCE OF THE SINGLE-SPAN TASK FORMULATION TO EXTRACTIVE QUESTIONANSWERING

Here, we introduce a newly compiled dataset consisting of questions with multiple
answers that originate from previously existing datasets. In addition, we run BERT-based models pretrained
for question-answering on our constructed dataset to evaluate their reading comprehension
abilities. Among the three of BERT-based models we ran, RoBERTa exhibits the highest consistent
performance, regardless of size. We find that all our models perform similarly on this new, multi-span
dataset (21.492% F1) compared to the single-span source datasets (~33.36% F1). While the models tested
on the source datasets were slightly fine-tuned, performance is similar enough to judge that task
formulation does not drastically affect question-answering abilities. Our evaluations indicate that these
models are indeed capable of adjusting to answer questions that require multiple answers. We hope that
our findings will assist future development in question-answering and improve existing question-answering
products and methods.
