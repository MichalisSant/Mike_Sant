Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> # ids_dense_model.py
... import numpy as np
... import pandas as pd
... import matplotlib.pyplot as plt
... import os
... 
... from sklearn.model_selection import train_test_split
... from sklearn.preprocessing import MinMaxScaler
... from sklearn.utils.class_weight import compute_class_weight
... from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
... 
... from tensorflow.keras.models import Sequential
... from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Input
... from tensorflow.keras.utils import to_categorical
... from tensorflow.keras.callbacks import EarlyStopping
... 
... # Επιλογή μοντέλου: True για Dense, False για Conv1D
... use_dense_model = True
... 
... # === ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ ===
... print("Φόρτωση δεδομένων...")
... url = "https://raw.githubusercontent.com/kdemertzis/EKPA/main/Data/pcap_data.csv"
... df = pd.read_csv(url)
... 
... # Αφαίρεση κατηγορίας που δε χρησιμοποιούμε
... df = df[df['Label'] != 'DrDoS_LDAP']
... df = df.sample(frac=1, random_state=42)
... 
... # Κωδικοποίηση labels
... label_map = {
...     'WebDDoS': 0, 'BENIGN': 1, 'UDP-lag': 2, 'DrDoS_NTP': 3,
...     'Syn': 4, 'DrDoS_SSDP': 5, 'DrDoS_UDP': 6, 'DrDoS_NetBIOS': 7,
...     'DrDoS_MSSQL': 8, 'DrDoS_SNMP': 9, 'TFTP': 10, 'DrDoS_DNS': 11
... }
... df['Label'] = df['Label'].map(label_map)
... 
# Επιλογή χαρακτηριστικών
selected_features = [38,47,37,48,11,9,7,52,10,36,1,34,4,17,19,57,21,
                     18,22,24,32,50,23,55,51,5,3,39,40,43,58,12,25,
                     20,2,35,67,33,6,53]

X = df.iloc[:, selected_features].values
y = df['Label'].values
n_classes = len(np.unique(y))

# Κανονικοποίηση
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Διαχωρισμός σε train, val, test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, stratify=y_train, random_state=42
)

# One-hot encoding
y_train_cat = to_categorical(y_train, num_classes=n_classes)
y_val_cat   = to_categorical(y_val, num_classes=n_classes)
y_test_cat  = to_categorical(y_test, num_classes=n_classes)

# Βάρη για μη ισορροπημένες κλάσεις
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights[i] for i in range(n_classes)}

# === ΟΡΙΣΜΟΣ ΜΟΝΤΕΛΩΝ ===

def model_conv1D(inshape, nclass):
    model = Sequential([
        Input(shape=(inshape, 1)),
        Conv1D(64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(nclass, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_dense(inshape, nclass):
    model = Sequential([
        Input(shape=(inshape,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(nclass, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Επιλογή μοντέλου
inshape = X_train.shape[1]
if use_dense_model:
    model = model_dense(inshape, n_classes)
else:
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    model = model_conv1D(inshape, n_classes)

# Εκτύπωση περίληψης μοντέλου
model.summary()

# === ΕΚΠΑΙΔΕΥΣΗ ===
print("Εκπαίδευση μοντέλου...")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=50,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)

# === ΑΞΙΟΛΟΓΗΣΗ ===
print("Αξιολόγηση...")
pred = model.predict(X_test)
pred_labels = np.argmax(pred, axis=1)

cm = confusion_matrix(y_test, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.keys()))
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# === ΓΡΑΦΗΜΑΤΑ ===
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.grid()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.grid()
