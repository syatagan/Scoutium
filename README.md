<h1>Business Problem</h1>
Predicting whether football players belong to the "average" or "highlighted" class based on the scores assigned to their characteristics by scouts.
<h1>Data Story</h1>
The dataset consists of information from Scoutium about football players observed in matches, their characteristics, evaluations by scouts, and the features scored by scouts during the match.7
<h3>Feature Lists</h3>
<p>
<h5>scoutium_attributes.csv contains following features</h5>
<table>
<tr><td>task_response_id</td><td>A scout's set of evaluations for all the players in a team's lineup during a match</td></tr>
<tr><td>match_id</td><td>id of match</td></tr>
<tr><td>evaluator_id</td><td>id of scout(evaluator)</td></tr>
<tr><td>player_id</td><td>id of player</td></tr>
<tr><td>position_id</td><td>id of position</td></tr>
<tr><td>analysis_id</td><td>id of analysis</td></tr>
<tr><td>attribute_id</td><td>id of attribte</td></tr>
<tr><td>attribute_value</td><td>evaluation score for specified attribute </td></tr>
</table>
<h5>scoutium_potential_labels.csv contains following features</h5>
<table>
<tr><td>task_response_id</td><td></td></tr>
<tr><td>match_id</td><td>id of match</td></tr>
<tr><td>evaluator_id</td><td>id of scout</td></tr>
<tr><td>player_id</td><td>id of player</td></tr>
<tr><td>potential_label</td><td>A label that indicates a scout's final decision regarding a player during a match.</td></tr>
</table>
<h1>Project Tasks</h1>
Step 1: Read the 'scoutium_attributes.csv' and 'scoutium_potential_labels.csv' files.</br>
Step 2: Merge the CSV files using the merge function (perform the merging based on the columns "task_response_id," "match_id," "evaluator_id," and "player_id").</br>
Step 3: Remove the "Goalkeeper" class (position_id = 1) from the dataset.</br>
Step 4: Remove the "below_average" class from the "potential_label" (the "below_average" class represents only 1% of the entire dataset).</br>
Step 5: Create a table using the "pivot_table" function from the modified dataset, where each row represents a player, and manipulate the data accordingly.</br>
Step 6: Use the Label Encoder function to numerically encode the categories in the "potential_label" column (average, highlighted).</br>
Step 7: Create a list named "num_cols" to store the numerical variable columns.</br>
Step 8: Apply StandardScaler to scale the data in all the columns saved in "num_cols."</br>
Step 9: Develop a machine learning model that predicts the potential labels of football players with the minimum error using the available dataset. Print metrics such as Roc_auc, f1, precision, recall, and accuracy.</br>
Step 10: Use the feature_importance function to determine the importance level of the variables and plot the ranking of the features.

