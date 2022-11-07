# AML project 1

Predict brain age from MRI features.

## Usage

Try to implement new features for subtasks 0 to 3 on the "develop" branch.
Do this by creating new functions in the files missing_values.py, outlier_detection.py and 
feature_selection.py. Then, you can test the different functions in the main.py file by simply 
changing the function used for each subtask.

```python
import missing_values as sub0
import outlier_detection as sub1
import feature_selection as sub2

# subtask 0: replace missing values
x_train, Y = sub0.fill_nan(x_train, y_train)

# subtask 1: outlier detection
x_train, Y = sub1.outlier_detection(x_train, Y)

# subtask 3: feature selection
x_smol, new_test = sub2.feature_select_tree(x_train, Y, test, 500)

```

## Merging
If a new function developed works, it can be merged into the main branch. Don't merge fast-forward.
