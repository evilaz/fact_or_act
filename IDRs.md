## Implementation Decision Records 

This files serves as documentation of choices and respective clarifications. Like ADRs but in a much smaller scope of project.

1. Why columns were treated differently:
- '**statement**' is obviously the main information and the focus, so more processing and more experimentation is explored here.
- '**subjects**' is actually a multi-valued categorical feature, representing a set of topics that the statement would belong to. The unique number of topics was small, therefore it made sense to apply multi-label one-hot-encoding
- '**party affiliation**' is a significant to the predicted target variable with fairly low cardinality, thus I chose one-hot encoding.
- rest columns were initially chosen to be concatenated into one column 'rest context' since they were more sparse and with high cardinality. 
But this approach is still to be further explored.
###
2. Why label was turned to a binary True/False like that:
- Simplification mostly but also something to experiment with.
###
3. Why there are 2 options for stop words:
- Because removing the standard list of stop words alters the text significantly and so I want to explore if that works better for this project or not.
