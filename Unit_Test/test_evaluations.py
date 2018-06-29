import sys
sys.path.insert(0, '../chem_scripts') # add file to be tested
sys.path.insert(0, '../modules')

import numpy as np
import pandas as pd
import evaluations
from formatting import cs_prep_data_y, cs_load_csv, cs_load_smiles
from prototype_user2 import f_nn
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, LambdaCallback
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint

def test_softmax():
	K = np.zeros((1,5))
	Y = evaluations.softmax(K)

	assert len(Y) == 1, 'length error'
	for i in np.arange(5):
		if Y[0][i] != 0.2:
			raise Exception('calculation error,set expected')


def test_cs_auc():
	X_test, y_test = cs_load_smiles('../data/tox_niehs_int_verytoxic_smiles.csv')
	y_tmp = y_test
	auc_train = evaluations.cs_auc(y_tmp, y_tmp)

	assert auc_train == 1,'auc calculation error'
	assert type(y_tmp) == np.ndarray, 'input dataset type error, set expected'


def test_cs_multiclass_auc():
	df = pd.read_csv("../data/tox_niehs_int_epa_rdkit.csv")
	y_true = pd.DataFrame([0, 1, 2, 1, 2, 0])
	y_pred = pd.DataFrame([1, 2, 0, 1, 2, 0])
	y_true, y_class = cs_prep_data_y(y_true, tasktype="classification")
	y_pred, y_class = cs_prep_data_y(y_pred, tasktype="classification")
	
	if y_class == 0:
		raise Exception("One-hot encoding fails!")
	total = evaluations.cs_multiclass_auc(y_true, y_pred)
	if total == float("inf") or total == float("-inf"):
		raise Exception("Broken function! The result cannot be infinit.")


def test_cs_compute_results():
	task = "epa"
	X, y = cs_load_csv("../data/tox_niehs_tv_"+task+"_rdkit.csv")

	X_train = X[0:500]
	y_train = y[0:500]
	X_valid = X[500:550]
	y_valid = y[500:550]
	X_test = X[550]
	y_test = y[550]

	y_train, y_class = cs_prep_data_y(y_train, tasktype='classification')   #ONLY DO THIS AFTER SPLITTING
	y_valid, y_class = cs_prep_data_y(y_valid, tasktype='classification')

	y, model = f_nn(task, 'mlp', pd.DataFrame([X_test]))
	df_out = pd.DataFrame(columns=['Train Loss', 'Validation Loss', 'Test Loss', 'Train RMSE', 'Validation RMSE', 'Test RMSE'])

	if df_out.shape[1] != 6:
		raise Exception("Columns number for result has to be exactly 6!")
	if y_class <= 0:
		raise Exception("Class should be at least 1.")
	if pd.DataFrame(X_train).empty:
		raise Exception("Training data cannot be empty!")
	if pd.DataFrame(X_valid).empty:
		raise Exception("Validation data cannot be empty!")
	if pd.DataFrame(X_test).empty:
		raise Exception("Test data cannot be empty!")
'''
	y_preds_test = evaluations.cs_compute_results(model, classes=y_class, df_out=df_out, train_data=(X_train,pd.DataFrame(y_train)), valid_data=(X_valid, pd.DataFrame(y_valid)), test_data=(X_test,y_test))
	
	if len(y_preds_test) == 0:
		raise Exception("Function is broken! The prediction cannot be zero!")
'''

def test_cs_keras_to_seaborn():
	task = "epa"
	X, y = cs_load_csv("../data/tox_niehs_tv_"+task+"_rdkit.csv")

	X_train = X[0:500]
	y_train = y[0:500]
	X_valid = X[500:550]
	y_valid = y[500:550]
	y_train, y_class = cs_prep_data_y(y_train, tasktype='classification')   #ONLY DO THIS AFTER SPLITTING
	y_valid, y_class = cs_prep_data_y(y_valid, tasktype='classification')
	y, model = f_nn(task, 'mlp', pd.DataFrame([X[550]]))

	batch_size = 128
	nb_epoch = 5
	verbose = 1
	filecp = "tox_niehs_mlp"+task+"_bestweights_trial_1_0.hdf5"
	filecsv = "tox_niehs_mlp_"+task+"_loss_curve_1_0.csv"
	callbacks = [TerminateOnNaN(),
                     LambdaCallback(on_epoch_end=lambda epoch,logs: sys.stdout.flush()),
                     EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='auto'),
                     ModelCheckpoint(filecp, monitor="val_loss", verbose=1, save_best_only=True, mode="auto"),
                     CSVLogger(filecsv)]
	hist = model.fit(x=X_train, y=y_train,
					batch_size=batch_size,
					epochs=nb_epoch,
					verbose=verbose,
					validation_data=(X_valid, y_valid),
					callbacks=callbacks)
	if pd.DataFrame(hist.history).empty:
		raise Exception("The fitting process is broken!")
	output_df = evaluations.cs_keras_to_seaborn(hist)
	if output_df.empty:
		raise Exception("Convert failure!")
