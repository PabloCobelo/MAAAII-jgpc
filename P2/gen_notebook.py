import json

def code_cell(source):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[source]}

def md_cell(source):
    return {"cell_type":"markdown","metadata":{},"source":[source]}

cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TÍTULO
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell(
    "# Práctica 2 — Aprendizaje Semi-Supervisado en CIFAR-100\n\n"
    "| Ejercicio | Técnica |\n"
    "|-----------|----------|\n"
    "| 1 | Clasificador supervisado (línea base) |\n"
    "| 2 | Auto-aprendizaje (*self-training*) |\n"
    "| 3 | Autoencoder en dos pasos |\n"
    "| 4 | Autoencoder en un paso (conjunto) |\n"
    "| 5 | Filtrado de anomalías + E2/E3/E4 |\n"
    "| 6 | Aprendizaje contrastivo (E3/E4/E5 equivalente) |\n\n"
    "**Partición:** 10 000 etiquetadas · 40 000 sin etiquetar · 10 000 test · 100 clases."
))

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell("## 0. Configuración global"))

cells.append(code_cell(
    "CFG = {\n"
    '    "seed": 42,\n'
    '    "frac_unlabeled": 0.80,\n'
    '    "val_size": 0.20,\n'
    '    "num_classes": 100,\n'
    '    "filters": [64, 128, 256],\n'
    '    "dense_units": 512,\n'
    '    "dropout": 0.2,\n'
    '    "l2_reg": 1e-4,\n'
    '    "lr": 3e-4,\n'
    '    "weight_decay": 1e-4,\n'
    '    "batch_size": 64,\n'
    '    "ae_batch_size": 128,\n'
    '    "epochs_baseline": 30,\n'
    '    "epochs_ae": 15,\n'
    '    "epochs_cls": 30,\n'
    '    "st_epochs": 30,\n'
    '    "st_iters": 5,\n'
    '    "st_threshold": 0.95,\n'
    '    "joint_alpha": 0.5,\n'
    '    "nu": 0.90,\n'
    '    "anomaly_epochs": 15,\n'
    '    "ad_delta": 0.025,\n'
    '    "ad_patience": 3,\n'
    '    "cl_tau": 5.0,\n'
    '    "cl_lambda": 0.5,\n'
    '    "cl_alpha": 0.5,\n'
    '    "cl_epochs": 8,\n'
    "}\n"
))

# ══════════════════════════════════════════════════════════════════════════════
# 1. IMPORTACIONES
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell("## 1. Importaciones"))

cells.append(code_cell(
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import tensorflow as tf\n"
    "from tensorflow.keras import regularizers\n"
    "from tensorflow.keras.layers import (\n"
    "    Input, Conv2D, MaxPooling2D, UpSampling2D,\n"
    "    Flatten, Dense, Dropout, BatchNormalization,\n"
    "    RandomRotation, RandomTranslation, RandomZoom, Resizing, RandomCrop,\n"
    ")\n"
    "from tensorflow.keras.models import Model\n"
    "from tensorflow.keras.utils import to_categorical\n"
    "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\n"
    "from tensorflow.keras.optimizers import AdamW, Adam\n"
    "from sklearn.model_selection import train_test_split\n"
    "\n"
    "SEED = CFG[\"seed\"]\n"
    "np.random.seed(SEED)\n"
    "tf.random.set_seed(SEED)\n"
    "print(f\"TensorFlow {tf.__version__} | Seed={SEED}\")\n"
))

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATOS
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell("## 2. Carga y preparación del dataset"))

cells.append(code_cell(
    "(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.cifar100.load_data()\n"
    "print(f\"Train: {x_train_raw.shape}  |  Test: {x_test_raw.shape}\")\n"
))

cells.append(code_cell(
    "num_train     = x_train_raw.shape[0]\n"
    "num_unlabeled = int(num_train * CFG[\"frac_unlabeled\"])\n"
    "num_labeled   = num_train - num_unlabeled\n"
    "\n"
    "rng = np.random.default_rng(SEED)\n"
    "idx = rng.permutation(num_train)\n"
    "labeled_idx, unlabeled_idx = idx[:num_labeled], idx[num_labeled:]\n"
    "\n"
    "x_labeled_raw = x_train_raw[labeled_idx]\n"
    "y_labeled_raw = y_train_raw[labeled_idx]\n"
    "\n"
    "x_train_s, x_val, y_train_raw_s, y_val_raw = train_test_split(\n"
    "    x_labeled_raw, y_labeled_raw,\n"
    "    test_size=CFG[\"val_size\"], random_state=SEED, stratify=y_labeled_raw\n"
    ")\n"
    "\n"
    "x_train     = x_train_s.astype(\"float32\") / 255.0\n"
    "x_val       = x_val.astype(\"float32\")     / 255.0\n"
    "x_test      = x_test_raw.astype(\"float32\") / 255.0\n"
    "x_unlabeled = x_train_raw[unlabeled_idx].astype(\"float32\") / 255.0\n"
    "\n"
    "y_train = to_categorical(y_train_raw_s.squeeze(), CFG[\"num_classes\"])\n"
    "y_val   = to_categorical(y_val_raw.squeeze(),     CFG[\"num_classes\"])\n"
    "y_test  = to_categorical(y_test_raw.squeeze(),    CFG[\"num_classes\"])\n"
    "\n"
    "print(f\"Train etiquetado : {x_train.shape[0]}\")\n"
    "print(f\"Validación       : {x_val.shape[0]}\")\n"
    "print(f\"Sin etiquetar    : {x_unlabeled.shape[0]}\")\n"
    "print(f\"Test             : {x_test.shape[0]}\")\n"
))

# EDA
cells.append(md_cell("### 2.1 Análisis exploratorio"))

cells.append(code_cell(
    "def plot_class_distribution(ys, titles, figsize=(14, 4)):\n"
    "    fig, axes = plt.subplots(1, len(ys), figsize=figsize, sharey=False)\n"
    "    for ax, y, title in zip(axes, ys, titles):\n"
    "        cls, cnt = np.unique(y.squeeze(), return_counts=True)\n"
    "        ax.bar(cls, cnt, color=\"steelblue\", width=1.0, edgecolor=\"none\")\n"
    "        ax.set_title(title)\n"
    "        ax.set_xlabel(\"Clase\")\n"
    "        ax.set_ylabel(\"Muestras\")\n"
    "        ax.grid(axis=\"y\", linewidth=0.5)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "\n"
    "plot_class_distribution(\n"
    "    [y_train_raw_s, y_val_raw, y_test_raw],\n"
    "    [\"Train etiquetado\", \"Validación\", \"Test\"],\n"
    ")\n"
    "\n"
    "fig, axes = plt.subplots(3, 8, figsize=(12, 5))\n"
    "for ax, img in zip(axes.flat, x_train[:24]):\n"
    "    ax.imshow(img)\n"
    "    ax.axis(\"off\")\n"
    "fig.suptitle(\"Muestra de imágenes de entrenamiento\")\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ══════════════════════════════════════════════════════════════════════════════
# 3. ARQUITECTURA COMPARTIDA
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell(
    "## 3. Arquitectura compartida\n\n"
    "El mismo bloque encoder se reutiliza en todos los ejercicios para garantizar comparabilidad."
))

cells.append(code_cell(
    "def build_conv_encoder(\n"
    "    input_shape=(32, 32, 3), filters=None, dropout=0.2, l2_reg=1e-4, name=\"encoder\"\n"
    ") -> Model:\n"
    "    \"\"\"\n"
    "    Encoder convolucional de 3 bloques: [Conv->BN->Conv->MaxPool->Dropout] x 3.\n"
    "    Cada bloque dobla el número de filtros respecto al anterior.\n"
    "    \"\"\"\n"
    "    if filters is None:\n"
    "        filters = [64, 128, 256]\n"
    "    inputs = Input(shape=input_shape)\n"
    "    x = inputs\n"
    "    for f in filters:\n"
    "        x = Conv2D(f, (3, 3), activation=\"relu\", padding=\"same\",\n"
    "                   kernel_regularizer=regularizers.l2(l2_reg))(x)\n"
    "        x = BatchNormalization()(x)\n"
    "        x = Conv2D(f, (3, 3), activation=\"relu\", padding=\"same\")(x)\n"
    "        x = MaxPooling2D((2, 2))(x)\n"
    "        x = Dropout(dropout)(x)\n"
    "    return Model(inputs, x, name=name)\n"
    "\n"
    "\n"
    "def build_decoder(encoded_shape, filters=None, name=\"decoder\") -> Model:\n"
    "    \"\"\"\n"
    "    Decoder simétrico al encoder usando UpSampling2D.\n"
    "    \"\"\"\n"
    "    if filters is None:\n"
    "        filters = [256, 128, 64]\n"
    "    enc_in = Input(shape=encoded_shape)\n"
    "    x = enc_in\n"
    "    for f in filters:\n"
    "        x = UpSampling2D((2, 2))(x)\n"
    "        x = Conv2D(f, (3, 3), activation=\"relu\", padding=\"same\")(x)\n"
    "        x = BatchNormalization()(x)\n"
    "    decoded = Conv2D(3, (3, 3), activation=\"sigmoid\", padding=\"same\")(x)\n"
    "    return Model(enc_in, decoded, name=name)\n"
    "\n"
    "\n"
    "def build_classifier_head(\n"
    "    encoder: Model, num_classes: int, dense_units: int = 512,\n"
    "    dropout: float = 0.2, l2_reg: float = 1e-4, name: str = \"classifier\"\n"
    ") -> Model:\n"
    "    \"\"\"\n"
    "    Cabeza clasificadora sobre un encoder Keras existente.\n"
    "    Arquitectura: Flatten -> Dense -> BN -> Dropout -> Dense(softmax)\n"
    "    \"\"\"\n"
    "    x = Flatten()(encoder.output)\n"
    "    x = Dense(dense_units, activation=\"relu\",\n"
    "              kernel_regularizer=regularizers.l2(l2_reg))(x)\n"
    "    x = BatchNormalization()(x)\n"
    "    x = Dropout(dropout)(x)\n"
    "    out = Dense(num_classes, activation=\"softmax\")(x)\n"
    "    return Model(encoder.input, out, name=name)\n"
    "\n"
    "\n"
    "def make_optimizer():\n"
    "    return AdamW(learning_rate=CFG[\"lr\"], weight_decay=CFG[\"weight_decay\"])\n"
))

# ══════════════════════════════════════════════════════════════════════════════
# 4. UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell("## 4. Funciones auxiliares"))

cells.append(code_cell(
    "def plot_training(history, title: str = \"\", figsize=(12, 4)):\n"
    "    fig, axes = plt.subplots(1, 2, figsize=figsize)\n"
    "    epochs = range(1, len(history.history[\"loss\"]) + 1)\n"
    "    axes[0].plot(epochs, history.history[\"loss\"],     label=\"Train\")\n"
    "    axes[0].plot(epochs, history.history[\"val_loss\"], label=\"Val\")\n"
    "    axes[0].set_title(f\"{title} — Pérdida\")\n"
    "    axes[0].set_xlabel(\"Época\")\n"
    "    axes[0].legend()\n"
    "    axes[0].grid(linewidth=0.5)\n"
    "    if \"accuracy\" in history.history:\n"
    "        axes[1].plot(epochs, history.history[\"accuracy\"],     label=\"Train\")\n"
    "        axes[1].plot(epochs, history.history[\"val_accuracy\"], label=\"Val\")\n"
    "        axes[1].set_title(f\"{title} — Accuracy\")\n"
    "        axes[1].set_xlabel(\"Época\")\n"
    "        axes[1].legend()\n"
    "        axes[1].grid(linewidth=0.5)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "\n"
    "\n"
    "def evaluate_and_report(model, x, y_onehot, name: str = \"Test\") -> dict:\n"
    "    loss, acc = model.evaluate(x, y_onehot, verbose=0)\n"
    "    print(f\"[{name}]  Loss: {loss:.4f}  |  Accuracy: {acc:.4f}\")\n"
    "    return {\"loss\": loss, \"accuracy\": acc}\n"
    "\n"
    "\n"
    "def plot_st_progress(accs: list, label: str = \"Accuracy en Test\"):\n"
    "    plt.figure(figsize=(7, 4))\n"
    "    plt.plot(range(1, len(accs) + 1), accs, \"o-\", color=\"steelblue\")\n"
    "    plt.xlabel(\"Iteración\")\n"
    "    plt.ylabel(\"Accuracy\")\n"
    "    plt.title(label)\n"
    "    plt.grid(linewidth=0.5)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "\n"
    "\n"
    "RESULTS = {}\n"
))

# ══════════════════════════════════════════════════════════════════════════════
# EJERCICIO 1
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell(
    "---\n"
    "## Ejercicio 1 — Clasificador supervisado (línea base)\n\n"
    "Modelo entrenado únicamente con las muestras etiquetadas."
))

cells.append(code_cell(
    "enc1     = build_conv_encoder(filters=CFG[\"filters\"], dropout=CFG[\"dropout\"],\n"
    "                               l2_reg=CFG[\"l2_reg\"], name=\"encoder_e1\")\n"
    "model_e1 = build_classifier_head(enc1, CFG[\"num_classes\"], CFG[\"dense_units\"],\n"
    "                                  CFG[\"dropout\"], CFG[\"l2_reg\"], name=\"classifier_e1\")\n"
    "model_e1.compile(optimizer=make_optimizer(), loss=\"categorical_crossentropy\",\n"
    "                 metrics=[\"accuracy\"])\n"
    "model_e1.summary(line_length=80)\n"
))

cells.append(code_cell(
    "history_e1 = model_e1.fit(\n"
    "    x_train, y_train,\n"
    "    epochs=CFG[\"epochs_baseline\"],\n"
    "    batch_size=CFG[\"batch_size\"],\n"
    "    validation_data=(x_val, y_val),\n"
    "    callbacks=[\n"
    "        EarlyStopping(monitor=\"val_accuracy\", patience=8, restore_best_weights=True),\n"
    "        ReduceLROnPlateau(monitor=\"val_loss\", factor=0.5, patience=4),\n"
    "    ],\n"
    "    verbose=1,\n"
    ")\n"
    "plot_training(history_e1, title=\"Ejercicio 1\")\n"
    "RESULTS[\"E1 Supervisado\"] = evaluate_and_report(model_e1, x_test, y_test, \"Test E1\")\n"
))

cells.append(md_cell(
    "### Ejercicio 1 — Preguntas\n\n"
    "**a.** ¿Qué red has escogido? ¿Por qué? ¿Cómo la has entrenado?\n\n"
    "**b.** ¿Cuál es el rendimiento del modelo en entrenamiento? ¿Y en prueba?\n\n"
    "**c.** ¿Qué conclusiones sacas de los resultados?"
))

# ══════════════════════════════════════════════════════════════════════════════
# EJERCICIO 2
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell(
    "---\n"
    "## Ejercicio 2 — Auto-aprendizaje (*Self-Training*)\n\n"
    "Incorpora las instancias no etiquetadas mediante pseudo-etiquetas ponderadas por confianza."
))

cells.append(code_cell(
    "def self_training(\n"
    "    x_labeled, y_labeled, x_unlabeled,\n"
    "    x_val_data, y_val_data, x_test_data, y_test_data,\n"
    "    n_iters=5, epochs_per_iter=20, threshold=0.95, batch_size=64,\n"
    ") -> tuple:\n"
    "    \"\"\"\n"
    "    Auto-aprendizaje iterativo con pseudo-etiquetas.\n"
    "\n"
    "    En cada iteración:\n"
    "      1. Se entrena un clasificador nuevo con los datos etiquetados acumulados.\n"
    "      2. Las predicciones con confianza >= threshold se convierten en\n"
    "         pseudo-etiquetas ponderadas por su confianza máxima.\n"
    "      3. Los datos pseudo-etiquetados se eliminan del pool sin etiquetar.\n"
    "\n"
    "    Returns:\n"
    "        (best_model, test_accuracies)\n"
    "    \"\"\"\n"
    "    train_x = x_labeled.copy()\n"
    "    train_y = y_labeled.copy()\n"
    "    pool_x  = x_unlabeled.copy()\n"
    "    weights = np.ones(len(train_y))\n"
    "    test_accs = []\n"
    "    best_acc, best_model = 0.0, None\n"
    "\n"
    "    for it in range(n_iters):\n"
    "        enc = build_conv_encoder(filters=CFG[\"filters\"], dropout=CFG[\"dropout\"],\n"
    "                                  l2_reg=CFG[\"l2_reg\"], name=f\"enc_st{it}\")\n"
    "        clf = build_classifier_head(enc, CFG[\"num_classes\"], CFG[\"dense_units\"],\n"
    "                                     CFG[\"dropout\"], CFG[\"l2_reg\"], name=f\"clf_st{it}\")\n"
    "        clf.compile(optimizer=make_optimizer(), loss=\"categorical_crossentropy\",\n"
    "                    metrics=[\"accuracy\"])\n"
    "        clf.fit(\n"
    "            train_x, train_y,\n"
    "            sample_weight=weights,\n"
    "            epochs=epochs_per_iter,\n"
    "            batch_size=batch_size,\n"
    "            validation_data=(x_val_data, y_val_data),\n"
    "            callbacks=[EarlyStopping(monitor=\"val_accuracy\", patience=6,\n"
    "                                     restore_best_weights=True)],\n"
    "            verbose=0,\n"
    "        )\n"
    "\n"
    "        _, test_acc = clf.evaluate(x_test_data, y_test_data, verbose=0)\n"
    "        test_accs.append(test_acc)\n"
    "        if test_acc > best_acc:\n"
    "            best_acc, best_model = test_acc, clf\n"
    "\n"
    "        if len(pool_x) == 0:\n"
    "            print(f\"Iter {it+1}: pool vacío.\")\n"
    "            break\n"
    "\n"
    "        proba = clf.predict(pool_x, verbose=0)\n"
    "        conf  = proba.max(axis=1)\n"
    "        pred  = proba.argmax(axis=1)\n"
    "        mask  = conf >= threshold\n"
    "        n_new = mask.sum()\n"
    "        if n_new > 0:\n"
    "            train_x = np.concatenate([train_x, pool_x[mask]])\n"
    "            train_y = np.concatenate([train_y,\n"
    "                                      to_categorical(pred[mask], CFG[\"num_classes\"])])\n"
    "            weights = np.concatenate([weights, conf[mask]])\n"
    "            pool_x  = pool_x[~mask]\n"
    "\n"
    "        print(f\"Iter {it+1}/{n_iters}  test={test_acc:.4f}  \"\n"
    "              f\"añadidos={n_new}  pool={len(pool_x)}\")\n"
    "        tf.keras.backend.clear_session()\n"
    "\n"
    "    return best_model, test_accs\n"
))

cells.append(code_cell(
    "best_model_e2, test_accs_e2 = self_training(\n"
    "    x_train, y_train, x_unlabeled,\n"
    "    x_val, y_val, x_test, y_test,\n"
    "    n_iters=CFG[\"st_iters\"],\n"
    "    epochs_per_iter=CFG[\"st_epochs\"],\n"
    "    threshold=CFG[\"st_threshold\"],\n"
    "    batch_size=CFG[\"batch_size\"],\n"
    ")\n"
    "plot_st_progress(test_accs_e2, \"E2 Self-Training — Accuracy en Test\")\n"
    "RESULTS[\"E2 Self-Training\"] = {\"accuracy\": max(test_accs_e2)}\n"
    "print(f\"Mejor accuracy en test (E2): {max(test_accs_e2):.4f}\")\n"
))

cells.append(md_cell(
    "### Ejercicio 2 — Preguntas\n\n"
    "**a.** ¿Qué parámetros has definido para el entrenamiento?\n\n"
    "**b.** ¿Cuál es el rendimiento del modelo en entrenamiento? ¿Y en prueba?\n\n"
    "**c.** ¿Se mejoran los resultados del Ejercicio 1?\n\n"
    "**d.** ¿Qué conclusiones sacas?"
))

# ══════════════════════════════════════════════════════════════════════════════
# EJERCICIO 3
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell(
    "---\n"
    "## Ejercicio 3 — Autoencoder en dos pasos\n\n"
    "**Paso 1:** Entrenar el autoencoder sobre todas las imágenes (MSE).\n\n"
    "**Paso 2:** Congelar el encoder y entrenar la cabeza clasificadora con datos etiquetados."
))

cells.append(code_cell(
    "def train_two_step_ae(x_unlabeled_pool, tag=\"e3\"):\n"
    "    \"\"\"\n"
    "    Entrena el autoencoder en dos pasos y devuelve (model, history_ae, history_cls).\n"
    "    El encoder del autoencoder tiene la misma arquitectura que E1/E2,\n"
    "    a excepción del último bloque (la cabeza clasificadora).\n"
    "    \"\"\"\n"
    "    # Paso 1: autoencoder sobre todos los datos\n"
    "    enc = build_conv_encoder(filters=CFG[\"filters\"], dropout=CFG[\"dropout\"],\n"
    "                              l2_reg=CFG[\"l2_reg\"], name=f\"encoder_{tag}\")\n"
    "    dec = build_decoder(enc.output_shape[1:],\n"
    "                         filters=list(reversed(CFG[\"filters\"])), name=f\"decoder_{tag}\")\n"
    "    ae_in  = Input(shape=(32, 32, 3))\n"
    "    ae_mdl = Model(ae_in, dec(enc(ae_in)), name=f\"autoencoder_{tag}\")\n"
    "    ae_mdl.compile(optimizer=Adam(learning_rate=CFG[\"lr\"]), loss=\"mse\")\n"
    "\n"
    "    x_ae = np.concatenate([x_train, x_unlabeled_pool], axis=0)\n"
    "    h_ae = ae_mdl.fit(\n"
    "        x_ae, x_ae,\n"
    "        epochs=CFG[\"epochs_ae\"],\n"
    "        batch_size=CFG[\"ae_batch_size\"],\n"
    "        validation_data=(x_val, x_val),\n"
    "        callbacks=[EarlyStopping(monitor=\"val_loss\", patience=5,\n"
    "                                 restore_best_weights=True)],\n"
    "        verbose=1,\n"
    "    )\n"
    "\n"
    "    # Paso 2: clasificador con encoder congelado\n"
    "    for layer in enc.layers:\n"
    "        layer.trainable = False\n"
    "    clf = build_classifier_head(enc, CFG[\"num_classes\"], CFG[\"dense_units\"],\n"
    "                                 CFG[\"dropout\"], CFG[\"l2_reg\"], name=f\"classifier_{tag}\")\n"
    "    clf.compile(optimizer=make_optimizer(), loss=\"categorical_crossentropy\",\n"
    "                metrics=[\"accuracy\"])\n"
    "    h_cls = clf.fit(\n"
    "        x_train, y_train,\n"
    "        epochs=CFG[\"epochs_cls\"],\n"
    "        batch_size=CFG[\"batch_size\"],\n"
    "        validation_data=(x_val, y_val),\n"
    "        callbacks=[\n"
    "            EarlyStopping(monitor=\"val_accuracy\", patience=8, restore_best_weights=True),\n"
    "            ReduceLROnPlateau(monitor=\"val_loss\", factor=0.5, patience=4),\n"
    "        ],\n"
    "        verbose=1,\n"
    "    )\n"
    "    return clf, ae_mdl, enc, h_ae, h_cls\n"
))

cells.append(code_cell(
    "clf_e3, ae_e3, enc_e3, h_ae_e3, h_cls_e3 = train_two_step_ae(x_unlabeled, tag=\"e3\")\n"
    "\n"
    "plot_training(h_ae_e3,  title=\"E3 — Autoencoder\")\n"
    "plot_training(h_cls_e3, title=\"E3 — Clasificador (encoder congelado)\")\n"
    "\n"
    "# Visualización de reconstrucciones\n"
    "n_show  = 8\n"
    "samples = x_val[:n_show]\n"
    "recons  = ae_e3.predict(samples, verbose=0)\n"
    "fig, axes = plt.subplots(2, n_show, figsize=(14, 3.5))\n"
    "for i in range(n_show):\n"
    "    axes[0, i].imshow(samples[i]);             axes[0, i].axis(\"off\")\n"
    "    axes[1, i].imshow(np.clip(recons[i],0,1)); axes[1, i].axis(\"off\")\n"
    "axes[0,0].set_title(\"Original\",      loc=\"left\", fontsize=9)\n"
    "axes[1,0].set_title(\"Reconstruida\",  loc=\"left\", fontsize=9)\n"
    "plt.suptitle(\"E3 — Reconstrucciones del autoencoder\")\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "RESULTS[\"E3 Autoencoder 2-step\"] = evaluate_and_report(clf_e3, x_test, y_test, \"Test E3\")\n"
))

cells.append(md_cell(
    "### Ejercicio 3 — Preguntas\n\n"
    "**a.** ¿Cuál es la arquitectura del modelo? ¿Y sus hiperparámetros?\n\n"
    "**b.** ¿Cuál es el rendimiento del modelo en entrenamiento? ¿Y en prueba?\n\n"
    "**c.** ¿Se mejoran los resultados de E1 y E2?\n\n"
    "**d.** ¿Qué conclusiones sacas?"
))

# ══════════════════════════════════════════════════════════════════════════════
# EJERCICIO 4
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell(
    "---\n"
    "## Ejercicio 4 — Autoencoder en un paso (entrenamiento conjunto)\n\n"
    "El encoder se entrena simultáneamente con:\n"
    "- **L_recon** (MSE): reconstrucción de todas las imágenes.\n"
    "- **L_cls** (entropía cruzada): clasificación de las etiquetadas.\n\n"
    "Pérdida total: `L = L_recon + α · L_cls`"
))

cells.append(code_cell(
    "class JointAutoencoderClassifier:\n"
    "    \"\"\"\n"
    "    Entrena autoencoder y clasificador en un único paso de gradiente.\n"
    "\n"
    "    La arquitectura del autoencoder es la misma que la del Ejercicio 3.\n"
    "    La combinación encoder+clasificador es idéntica al Ejercicio 1.\n"
    "    \"\"\"\n"
    "\n"
    "    def __init__(self, input_shape=(32,32,3), num_classes=100,\n"
    "                 filters=None, dense_units=512, dropout=0.2,\n"
    "                 l2_reg=1e-4, alpha=0.5, lr=3e-4, weight_decay=1e-4):\n"
    "        if filters is None:\n"
    "            filters = [64, 128, 256]\n"
    "        self.alpha = alpha\n"
    "        self.optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)\n"
    "\n"
    "        # Encoder compartido (misma arquitectura que E1/E3)\n"
    "        self.encoder = build_conv_encoder(input_shape, filters, dropout, l2_reg, \"enc_e4\")\n"
    "        # Decoder (misma arquitectura que E3)\n"
    "        self.decoder = build_decoder(self.encoder.output_shape[1:],\n"
    "                                     list(reversed(filters)), \"dec_e4\")\n"
    "        # Cabeza clasificadora (misma que E1)\n"
    "        x = Flatten()(self.encoder.output)\n"
    "        x = Dense(dense_units, activation=\"relu\",\n"
    "                  kernel_regularizer=regularizers.l2(l2_reg))(x)\n"
    "        x = BatchNormalization()(x)\n"
    "        x = Dropout(dropout)(x)\n"
    "        out = Dense(num_classes, activation=\"softmax\")(x)\n"
    "        self.cls_head = Model(self.encoder.input, out, name=\"cls_head_e4\")\n"
    "\n"
    "        self._mse = tf.keras.losses.MeanSquaredError()\n"
    "        self._cce = tf.keras.losses.CategoricalCrossentropy()\n"
    "\n"
    "    @tf.function\n"
    "    def _labeled_step(self, x_b, y_b):\n"
    "        with tf.GradientTape() as tape:\n"
    "            enc_out = self.encoder(x_b, training=True)\n"
    "            loss = (self._mse(x_b, self.decoder(enc_out, training=True))\n"
    "                    + self.alpha * self._cce(y_b, self.cls_head(x_b, training=True)))\n"
    "        all_vars = (self.encoder.trainable_variables\n"
    "                    + self.decoder.trainable_variables\n"
    "                    + self.cls_head.trainable_variables)\n"
    "        self.optimizer.apply_gradients(zip(tape.gradient(loss, all_vars), all_vars))\n"
    "        return loss\n"
    "\n"
    "    @tf.function\n"
    "    def _unlabeled_step(self, x_b):\n"
    "        with tf.GradientTape() as tape:\n"
    "            loss = self._mse(x_b, self.decoder(\n"
    "                self.encoder(x_b, training=True), training=True))\n"
    "        vars_ = self.encoder.trainable_variables + self.decoder.trainable_variables\n"
    "        self.optimizer.apply_gradients(zip(tape.gradient(loss, vars_), vars_))\n"
    "        return loss\n"
    "\n"
    "    def fit(self, x_labeled, y_labeled, x_unlabeled_pool, x_val, y_val,\n"
    "            epochs=15, batch_size=64):\n"
    "        \"\"\"\n"
    "        Entrenamiento conjunto sobre datos etiquetados y sin etiquetar.\n"
    "        Returns historial con 'total_loss' y 'val_accuracy'.\n"
    "        \"\"\"\n"
    "        ds_l = (tf.data.Dataset.from_tensor_slices((x_labeled, y_labeled))\n"
    "                .shuffle(len(x_labeled), seed=SEED).batch(batch_size)\n"
    "                .prefetch(tf.data.AUTOTUNE))\n"
    "        ds_u = (tf.data.Dataset.from_tensor_slices(x_unlabeled_pool)\n"
    "                .shuffle(len(x_unlabeled_pool), seed=SEED).batch(batch_size)\n"
    "                .prefetch(tf.data.AUTOTUNE))\n"
    "        history = {\"total_loss\": [], \"val_accuracy\": []}\n"
    "        for ep in range(epochs):\n"
    "            losses = [float(self._labeled_step(xb, yb)) for xb, yb in ds_l]\n"
    "            for xb in ds_u:\n"
    "                self._unlabeled_step(xb)\n"
    "            val_acc = float(np.mean(\n"
    "                self.cls_head.predict(x_val, verbose=0).argmax(1) == y_val.argmax(1)\n"
    "            ))\n"
    "            history[\"total_loss\"].append(np.mean(losses))\n"
    "            history[\"val_accuracy\"].append(val_acc)\n"
    "            print(f\"Época {ep+1}/{epochs}  loss={np.mean(losses):.4f}  \"\n"
    "                  f\"val_acc={val_acc:.4f}\")\n"
    "        return history\n"
    "\n"
    "    def evaluate(self, x, y):\n"
    "        preds = self.cls_head.predict(x, verbose=0)\n"
    "        return float(self._cce(y, preds)), float(np.mean(preds.argmax(1) == y.argmax(1)))\n"
    "\n"
    "\n"
    "def train_joint_ae(x_unlabeled_pool, tag=\"e4\"):\n"
    "    \"\"\"Wrapper para entrenar y evaluar el modelo conjunto. Devuelve (model, history).\"\"\"\n"
    "    mdl = JointAutoencoderClassifier(\n"
    "        num_classes=CFG[\"num_classes\"], filters=CFG[\"filters\"],\n"
    "        dense_units=CFG[\"dense_units\"], dropout=CFG[\"dropout\"],\n"
    "        l2_reg=CFG[\"l2_reg\"], alpha=CFG[\"joint_alpha\"],\n"
    "        lr=CFG[\"lr\"], weight_decay=CFG[\"weight_decay\"],\n"
    "    )\n"
    "    h = mdl.fit(x_train, y_train, x_unlabeled_pool, x_val, y_val,\n"
    "                epochs=CFG[\"epochs_ae\"], batch_size=CFG[\"ae_batch_size\"])\n"
    "    return mdl, h\n"
))

cells.append(code_cell(
    "joint_e4, history_e4 = train_joint_ae(x_unlabeled, tag=\"e4\")\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n"
    "ep = range(1, len(history_e4[\"total_loss\"]) + 1)\n"
    "ax1.plot(ep, history_e4[\"total_loss\"], \"o-\")\n"
    "ax1.set_title(\"E4 — Pérdida conjunta\")\n"
    "ax1.set_xlabel(\"Época\")\n"
    "ax1.grid(linewidth=0.5)\n"
    "ax2.plot(ep, history_e4[\"val_accuracy\"], \"o-\", color=\"green\")\n"
    "ax2.set_title(\"E4 — Accuracy en validación\")\n"
    "ax2.set_xlabel(\"Época\")\n"
    "ax2.grid(linewidth=0.5)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "_, test_acc_e4 = joint_e4.evaluate(x_test, y_test)\n"
    "print(f\"[Test E4]  Accuracy: {test_acc_e4:.4f}\")\n"
    "RESULTS[\"E4 Autoencoder 1-step\"] = {\"accuracy\": test_acc_e4}\n"
))

cells.append(md_cell(
    "### Ejercicio 4 — Preguntas\n\n"
    "**a.** ¿Cuál es la arquitectura del modelo? ¿Y sus hiperparámetros?\n\n"
    "**b.** ¿Cuál es el rendimiento del modelo en entrenamiento? ¿Y en prueba?\n\n"
    "**c.** ¿Se mejoran los resultados de los ejercicios anteriores?\n\n"
    "**d.** ¿Qué conclusiones sacas?"
))

# ══════════════════════════════════════════════════════════════════════════════
# EJERCICIO 5 — Detector de anomalías
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell(
    "---\n"
    "## Ejercicio 5 — Filtrado de anomalías + E2/E3/E4\n\n"
    "Se entrena un detector de anomalías de una clase sobre los datos etiquetados (ν = 0.9).\n"
    "Las muestras más atípicas del pool sin etiquetar se descartan antes de aplicar E2, E3 y E4."
))

cells.append(code_cell(
    "class AdaptiveRCallback(tf.keras.callbacks.Callback):\n"
    "    \"\"\"\n"
    "    Actualiza el radio `r` del detector SVDD al final de cada época y detecta convergencia.\n"
    "    \"\"\"\n"
    "\n"
    "    def __init__(self, train_data, delta=0.025, patience=3):\n"
    "        super().__init__()\n"
    "        self.train_data = train_data\n"
    "        self.delta      = delta\n"
    "        self.patience   = patience\n"
    "        self._no_change = 0\n"
    "\n"
    "    def on_epoch_end(self, epoch, logs=None):\n"
    "        scores = self.model.predict(self.train_data, verbose=0).flatten()\n"
    "        new_r  = float(np.sort(scores)[int(len(scores) * (1.0 - self.model.nu))])\n"
    "        old_r  = float(self.model.r.numpy())\n"
    "        self.model.r.assign(new_r)\n"
    "        if abs(new_r - old_r) < self.delta:\n"
    "            self._no_change += 1\n"
    "            if self._no_change >= self.patience:\n"
    "                print(f\"  [AdaptiveR] Convergencia en época {epoch+1}.\")\n"
    "                self.model.stop_training = True\n"
    "        else:\n"
    "            self._no_change = 0\n"
    "\n"
    "\n"
    "class OneClassDetector:\n"
    "    \"\"\"\n"
    "    Detector de anomalías neuronal (variante SVDD).\n"
    "\n"
    "    Arquitectura idéntica al clasificador del Ejercicio 1,\n"
    "    a excepción de la capa de salida (Dense(1) lineal en lugar de softmax).\n"
    "    Estructura: [Conv->BN->Conv->MaxPool->Dropout]x3 -> Flatten -> Dense(512) -> BN -> Dropout -> Dense(1)\n"
    "    \"\"\"\n"
    "\n"
    "    def __init__(self, input_shape=(32, 32, 3), nu=0.9):\n"
    "        self.nu = nu\n"
    "        # Encoder idéntico al de E1 (misma función build_conv_encoder)\n"
    "        enc = build_conv_encoder(input_shape, CFG[\"filters\"], CFG[\"dropout\"],\n"
    "                                  CFG[\"l2_reg\"], name=\"enc_detector\")\n"
    "        # Cabeza: igual que E1 pero con salida lineal escalar\n"
    "        x = Flatten()(enc.output)\n"
    "        x = Dense(CFG[\"dense_units\"], activation=\"relu\",\n"
    "                  kernel_regularizer=regularizers.l2(CFG[\"l2_reg\"]))(x)\n"
    "        x = BatchNormalization()(x)\n"
    "        x = Dropout(CFG[\"dropout\"])(x)\n"
    "        outputs = Dense(1, name=\"score\")(x)  # salida lineal (sin activación)\n"
    "\n"
    "        self.model = tf.keras.Model(enc.input, outputs)\n"
    "        self.model.r  = tf.Variable(1.0, trainable=False)\n"
    "        self.model.nu = nu\n"
    "        self.model.compile(optimizer=\"adam\", loss=self._svdd_loss)\n"
    "\n"
    "    def _svdd_loss(self, y_true, y_pred):\n"
    "        \"\"\"Pérdida SVDD: penaliza puntuaciones por debajo del radio r.\"\"\"\n"
    "        return (1.0 / self.model.nu) * tf.reduce_mean(\n"
    "            tf.maximum(0.0, self.model.r - y_pred)\n"
    "        )\n"
    "\n"
    "    def fit(self, X, epochs=15, batch_size=128):\n"
    "        cb = AdaptiveRCallback(X, delta=CFG[\"ad_delta\"], patience=CFG[\"ad_patience\"])\n"
    "        self.model.fit(X, np.zeros((len(X), 1), dtype=\"float32\"),\n"
    "                       epochs=epochs, batch_size=batch_size, callbacks=[cb], verbose=1)\n"
    "\n"
    "    def filter_inliers(self, X, percentile=None):\n"
    "        \"\"\"\n"
    "        Devuelve el subconjunto de X considerado normal (inliers).\n"
    "        Descarta el (1-nu)*100 % inferior por puntuación.\n"
    "        \"\"\"\n"
    "        q      = percentile if percentile is not None else (1.0 - self.nu)\n"
    "        scores = self.model.predict(X, verbose=0).flatten()\n"
    "        thr    = np.quantile(scores, q)\n"
    "        mask   = scores > thr\n"
    "        print(f\"  Filtrado: {mask.sum()}/{len(X)} inliers ({100*mask.mean():.1f}%)\")\n"
    "        return X[mask]\n"
))

cells.append(code_cell(
    "# Entrenar detector con nu=0.9 sobre datos etiquetados (considerados normales)\n"
    "detector_e5 = OneClassDetector(nu=CFG[\"nu\"])\n"
    "detector_e5.fit(x_train, epochs=CFG[\"anomaly_epochs\"])\n"
    "\n"
    "# Filtrar pool sin etiquetar\n"
    "x_unlabeled_clean = detector_e5.filter_inliers(x_unlabeled)\n"
))

# E5.2 — ST con datos filtrados
cells.append(md_cell("### Ejercicio 5.2 — Auto-aprendizaje con datos filtrados"))

cells.append(code_cell(
    "best_model_e5_st, test_accs_e5_st = self_training(\n"
    "    x_train, y_train, x_unlabeled_clean,\n"
    "    x_val, y_val, x_test, y_test,\n"
    "    n_iters=CFG[\"st_iters\"],\n"
    "    epochs_per_iter=CFG[\"st_epochs\"],\n"
    "    threshold=CFG[\"st_threshold\"],\n"
    "    batch_size=CFG[\"batch_size\"],\n"
    ")\n"
    "plot_st_progress(test_accs_e5_st, \"E5.2 Anomaly+ST — Accuracy en Test\")\n"
    "RESULTS[\"E5.2 Anomaly+ST\"] = {\"accuracy\": max(test_accs_e5_st)}\n"
    "print(f\"Mejor accuracy E5.2: {max(test_accs_e5_st):.4f}\")\n"
))

# E5.3 — Autoencoder 2 pasos con datos filtrados
cells.append(md_cell("### Ejercicio 5.3 — Autoencoder dos pasos con datos filtrados"))

cells.append(code_cell(
    "clf_e5_3, ae_e5_3, enc_e5_3, h_ae_e5_3, h_cls_e5_3 = train_two_step_ae(\n"
    "    x_unlabeled_clean, tag=\"e5_3\"\n"
    ")\n"
    "plot_training(h_ae_e5_3,  title=\"E5.3 — Autoencoder (datos filtrados)\")\n"
    "plot_training(h_cls_e5_3, title=\"E5.3 — Clasificador\")\n"
    "RESULTS[\"E5.3 Anomaly+AE2step\"] = evaluate_and_report(clf_e5_3, x_test, y_test, \"Test E5.3\")\n"
))

# E5.4 — Autoencoder 1 paso con datos filtrados
cells.append(md_cell("### Ejercicio 5.4 — Autoencoder un paso con datos filtrados"))

cells.append(code_cell(
    "joint_e5_4, history_e5_4 = train_joint_ae(x_unlabeled_clean, tag=\"e5_4\")\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n"
    "ep = range(1, len(history_e5_4[\"total_loss\"]) + 1)\n"
    "ax1.plot(ep, history_e5_4[\"total_loss\"], \"o-\")\n"
    "ax1.set_title(\"E5.4 — Pérdida conjunta (filtrado)\")\n"
    "ax1.set_xlabel(\"Época\")\n"
    "ax1.grid(linewidth=0.5)\n"
    "ax2.plot(ep, history_e5_4[\"val_accuracy\"], \"o-\", color=\"green\")\n"
    "ax2.set_title(\"E5.4 — Accuracy en validación\")\n"
    "ax2.set_xlabel(\"Época\")\n"
    "ax2.grid(linewidth=0.5)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "_, test_acc_e5_4 = joint_e5_4.evaluate(x_test, y_test)\n"
    "print(f\"[Test E5.4]  Accuracy: {test_acc_e5_4:.4f}\")\n"
    "RESULTS[\"E5.4 Anomaly+AE1step\"] = {\"accuracy\": test_acc_e5_4}\n"
))

cells.append(md_cell(
    "### Ejercicio 5 — Preguntas\n\n"
    "**a.** ¿Se mejoran los resultados con respecto a los ejercicios anteriores? ¿Qué conclusiones sacas?"
))

# ══════════════════════════════════════════════════════════════════════════════
# EJERCICIO 6 — Aprendizaje contrastivo
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell(
    "---\n"
    "## Ejercicio 6 — Aprendizaje contrastivo (\"Hay vida más allá del autoencoder\")\n\n"
    "Se reemplazan los autoencoders de E3-E5 por la técnica contrastiva del Notebook 4:\n"
    "pérdida de consistencia (InfoNCE) + pérdida de clúster.\n\n"
    "- **E6.3** equivale a E3 (pre-entrenamiento + clasificador, todos los datos).\n"
    "- **E6.4** equivale a E4 (entrenamiento conjunto: contrastivo + clasificación).\n"
    "- **E6.5** equivale a E5.3 (filtrado de anomalías + pre-entrenamiento contrastivo)."
))

# Encoder subclase reutilizable
cells.append(code_cell(
    "class ConvEncoderSubclass(tf.keras.Model):\n"
    "    \"\"\"\n"
    "    Encoder convolucional como tf.keras.Model subclass.\n"
    "    Arquitectura idéntica a build_conv_encoder para uso con GradientTape.\n"
    "    \"\"\"\n"
    "\n"
    "    def __init__(self, filters=None, dropout=0.2):\n"
    "        super().__init__(name=\"conv_encoder_cl\")\n"
    "        if filters is None:\n"
    "            filters = [64, 128, 256]\n"
    "        self._layer_list = []\n"
    "        for f in filters:\n"
    "            self._layer_list += [\n"
    "                Conv2D(f, (3, 3), activation=\"relu\", padding=\"same\"),\n"
    "                BatchNormalization(),\n"
    "                Conv2D(f, (3, 3), activation=\"relu\", padding=\"same\"),\n"
    "                MaxPooling2D((2, 2)),\n"
    "                Dropout(dropout),\n"
    "            ]\n"
    "\n"
    "    def call(self, inputs, training=False):\n"
    "        x = inputs\n"
    "        for layer in self._layer_list:\n"
    "            x = (layer(x, training=training)\n"
    "                 if isinstance(layer, (Dropout, BatchNormalization)) else layer(x))\n"
    "        return x\n"
    "\n"
    "\n"
    "def build_cl_classifier(feat_dim, num_classes=100, dense_units=512,\n"
    "                         dropout=0.2, l2_reg=1e-4):\n"
    "    \"\"\"Cabeza clasificadora lineal sobre representaciones contrastivas.\"\"\"\n"
    "    inp = Input(shape=(feat_dim,))\n"
    "    x   = Dense(dense_units, activation=\"relu\",\n"
    "                kernel_regularizer=regularizers.l2(l2_reg))(inp)\n"
    "    x   = BatchNormalization()(x)\n"
    "    x   = Dropout(dropout)(x)\n"
    "    out = Dense(num_classes, activation=\"softmax\")(x)\n"
    "    return Model(inp, out, name=\"classifier_cl\")\n"
    "\n"
    "\n"
    "def extract_features(encoder_model, X, batch_size=256):\n"
    "    \"\"\"Extrae y aplana representaciones del encoder en lotes.\"\"\"\n"
    "    feats = []\n"
    "    for i in range(0, len(X), batch_size):\n"
    "        z = encoder_model(X[i:i+batch_size], training=False)\n"
    "        feats.append(tf.reshape(z, (tf.shape(z)[0], -1)).numpy())\n"
    "    return np.concatenate(feats, axis=0)\n"
))

# ContrastivePretrainer
cells.append(code_cell(
    "class ContrastivePretrainer:\n"
    "    \"\"\"\n"
    "    Pre-entrenamiento auto-supervisado con pérdida InfoNCE + pérdida de clúster.\n"
    "\n"
    "    Pérdida total: L = L_InfoNCE + lambda · L_cluster\n"
    "\n"
    "    Args:\n"
    "        encoder: ConvEncoderSubclass.\n"
    "        num_clusters: Número de clústeres (igual al número de clases).\n"
    "        tau: Temperatura de InfoNCE.\n"
    "        lambda_: Peso de la pérdida de clúster.\n"
    "        lr: Learning rate.\n"
    "    \"\"\"\n"
    "\n"
    "    def __init__(self, encoder, num_clusters=100, tau=5.0, lambda_=0.5, lr=3e-4):\n"
    "        self.encoder      = encoder\n"
    "        self.cluster_head = Dense(num_clusters, activation=\"softmax\")\n"
    "        self.tau          = tau\n"
    "        self.lambda_      = lambda_\n"
    "        self.optimizer    = Adam(learning_rate=lr)\n"
    "        self.aug1 = tf.keras.Sequential([\n"
    "            RandomRotation(0.05), RandomTranslation(0.15, 0.15), RandomZoom(0.15)\n"
    "        ])\n"
    "        self.aug2 = tf.keras.Sequential([\n"
    "            RandomTranslation(0.15, 0.15), Resizing(48, 48), RandomCrop(32, 32)\n"
    "        ])\n"
    "\n"
    "    def _encode_flat(self, x, training=False):\n"
    "        z = self.encoder(x, training=training)\n"
    "        return tf.reshape(z, (tf.shape(z)[0], -1))\n"
    "\n"
    "    @tf.function\n"
    "    def _train_step(self, x_batch):\n"
    "        with tf.GradientTape() as tape:\n"
    "            z1 = self._encode_flat(self.aug1(x_batch, training=True), training=True)\n"
    "            z2 = self._encode_flat(self.aug2(x_batch, training=True), training=True)\n"
    "            n  = tf.shape(x_batch)[0]\n"
    "            logits = tf.matmul(z1, z2, transpose_b=True) / self.tau\n"
    "            loss_m = tf.reduce_mean(\n"
    "                tf.keras.losses.sparse_categorical_crossentropy(\n"
    "                    tf.range(n), logits[:n, :n], from_logits=True\n"
    "                )\n"
    "            )\n"
    "            c1, c2 = self.cluster_head(z1), self.cluster_head(z2)\n"
    "            loss_c = tf.reduce_mean(\n"
    "                tf.reduce_sum(c1 * (1 - c1), axis=1)\n"
    "                + tf.reduce_sum(c2 * (1 - c2), axis=1)\n"
    "            )\n"
    "            total = loss_m + self.lambda_ * loss_c\n"
    "        vars_ = self.encoder.trainable_variables + self.cluster_head.trainable_variables\n"
    "        self.optimizer.apply_gradients(zip(tape.gradient(total, vars_), vars_))\n"
    "        return total\n"
    "\n"
    "    def fit(self, X, epochs=8, batch_size=128):\n"
    "        \"\"\"Pre-entrenamiento auto-supervisado sobre X.\"\"\"\n"
    "        ds = (tf.data.Dataset.from_tensor_slices(X)\n"
    "              .shuffle(len(X), seed=SEED)\n"
    "              .batch(batch_size, drop_remainder=True)\n"
    "              .prefetch(tf.data.AUTOTUNE))\n"
    "        losses = []\n"
    "        for ep in range(epochs):\n"
    "            bl = [float(self._train_step(xb)) for xb in ds]\n"
    "            losses.append(np.mean(bl))\n"
    "            print(f\"Época {ep+1}/{epochs}  loss={losses[-1]:.4f}\")\n"
    "        return losses\n"
))

# E6.3
cells.append(md_cell("### Ejercicio 6.3 — Pre-entrenamiento contrastivo + clasificador (todos los datos)"))

cells.append(code_cell(
    "enc_e6_3 = ConvEncoderSubclass(filters=CFG[\"filters\"], dropout=CFG[\"dropout\"])\n"
    "_ = enc_e6_3(tf.zeros((1, 32, 32, 3)))  # inicializar pesos\n"
    "\n"
    "pt_e6_3  = ContrastivePretrainer(enc_e6_3, num_clusters=CFG[\"num_classes\"],\n"
    "                                  tau=CFG[\"cl_tau\"], lambda_=CFG[\"cl_lambda\"], lr=CFG[\"lr\"])\n"
    "x_all_e6 = np.concatenate([x_train, x_unlabeled], axis=0)\n"
    "cl_losses_e6_3 = pt_e6_3.fit(x_all_e6, epochs=CFG[\"cl_epochs\"])\n"
    "\n"
    "plt.figure(figsize=(7, 4))\n"
    "plt.plot(range(1, len(cl_losses_e6_3)+1), cl_losses_e6_3, \"o-\")\n"
    "plt.title(\"E6.3 — Pérdida contrastiva\")\n"
    "plt.xlabel(\"Época\")\n"
    "plt.ylabel(\"Loss\")\n"
    "plt.grid(linewidth=0.5)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "# Extraer representaciones y entrenar clasificador\n"
    "z_train_63 = extract_features(enc_e6_3, x_train)\n"
    "z_val_63   = extract_features(enc_e6_3, x_val)\n"
    "z_test_63  = extract_features(enc_e6_3, x_test)\n"
    "\n"
    "clf_e6_3 = build_cl_classifier(z_train_63.shape[1], CFG[\"num_classes\"],\n"
    "                                CFG[\"dense_units\"], CFG[\"dropout\"], CFG[\"l2_reg\"])\n"
    "clf_e6_3.compile(optimizer=make_optimizer(), loss=\"categorical_crossentropy\",\n"
    "                 metrics=[\"accuracy\"])\n"
    "h_cls_e6_3 = clf_e6_3.fit(\n"
    "    z_train_63, y_train,\n"
    "    epochs=CFG[\"epochs_cls\"],\n"
    "    batch_size=CFG[\"batch_size\"],\n"
    "    validation_data=(z_val_63, y_val),\n"
    "    callbacks=[\n"
    "        EarlyStopping(monitor=\"val_accuracy\", patience=8, restore_best_weights=True),\n"
    "        ReduceLROnPlateau(monitor=\"val_loss\", factor=0.5, patience=4),\n"
    "    ],\n"
    "    verbose=1,\n"
    ")\n"
    "plot_training(h_cls_e6_3, title=\"E6.3 — Clasificador contrastivo\")\n"
    "RESULTS[\"E6.3 Contrastivo\"] = evaluate_and_report(clf_e6_3, z_test_63, y_test, \"Test E6.3\")\n"
))

# E6.4 — Joint contrastivo + clasificación
cells.append(md_cell(
    "### Ejercicio 6.4 — Entrenamiento conjunto contrastivo + clasificador\n\n"
    "Equivalente a E4: encoder entrenado simultáneamente con pérdida contrastiva (todos los datos)\n"
    "y pérdida de clasificación (datos etiquetados)."
))

cells.append(code_cell(
    "class JointContrastiveClassifier:\n"
    "    \"\"\"\n"
    "    Entrena el encoder con pérdida contrastiva (InfoNCE + clúster) sobre todos los datos\n"
    "    y pérdida de clasificación sobre los datos etiquetados, en un único paso de gradiente.\n"
    "\n"
    "    La arquitectura del encoder es la misma que la del Ejercicio 6.3 (= E1/E3).\n"
    "    \"\"\"\n"
    "\n"
    "    def __init__(self, num_classes=100, num_clusters=100, filters=None, dense_units=512,\n"
    "                 dropout=0.2, l2_reg=1e-4, tau=5.0, cl_lambda=0.5, alpha=0.5, lr=3e-4):\n"
    "        if filters is None:\n"
    "            filters = [64, 128, 256]\n"
    "        self.tau    = tau\n"
    "        self.lambda_ = cl_lambda\n"
    "        self.alpha   = alpha\n"
    "        self.optimizer = Adam(learning_rate=lr)\n"
    "\n"
    "        self.encoder      = ConvEncoderSubclass(filters=filters, dropout=dropout)\n"
    "        self.cluster_head = Dense(num_clusters, activation=\"softmax\")\n"
    "        self._cce         = tf.keras.losses.CategoricalCrossentropy()\n"
    "\n"
    "        # La cabeza clasificadora se construye tras el primer forward pass\n"
    "        self._dense_units = dense_units\n"
    "        self._dropout     = dropout\n"
    "        self._l2_reg      = l2_reg\n"
    "        self._num_classes = num_classes\n"
    "        self._cls_layers  = [\n"
    "            Dense(dense_units, activation=\"relu\",\n"
    "                  kernel_regularizer=regularizers.l2(l2_reg)),\n"
    "            BatchNormalization(),\n"
    "            Dropout(dropout),\n"
    "            Dense(num_classes, activation=\"softmax\"),\n"
    "        ]\n"
    "\n"
    "        self.aug1 = tf.keras.Sequential([\n"
    "            RandomRotation(0.05), RandomTranslation(0.15, 0.15), RandomZoom(0.15)\n"
    "        ])\n"
    "        self.aug2 = tf.keras.Sequential([\n"
    "            RandomTranslation(0.15, 0.15), Resizing(48, 48), RandomCrop(32, 32)\n"
    "        ])\n"
    "\n"
    "    def _encode_flat(self, x, training=False):\n"
    "        z = self.encoder(x, training=training)\n"
    "        return tf.reshape(z, (tf.shape(z)[0], -1))\n"
    "\n"
    "    def _cls_forward(self, z, training=False):\n"
    "        x = z\n"
    "        for layer in self._cls_layers:\n"
    "            x = (layer(x, training=training)\n"
    "                 if isinstance(layer, (Dropout, BatchNormalization)) else layer(x))\n"
    "        return x\n"
    "\n"
    "    @tf.function\n"
    "    def _train_step(self, x_all, x_lab, y_lab):\n"
    "        with tf.GradientTape() as tape:\n"
    "            # Pérdida contrastiva sobre todos los datos\n"
    "            z1 = self._encode_flat(self.aug1(x_all, training=True), training=True)\n"
    "            z2 = self._encode_flat(self.aug2(x_all, training=True), training=True)\n"
    "            n  = tf.shape(x_all)[0]\n"
    "            logits = tf.matmul(z1, z2, transpose_b=True) / self.tau\n"
    "            loss_m = tf.reduce_mean(\n"
    "                tf.keras.losses.sparse_categorical_crossentropy(\n"
    "                    tf.range(n), logits[:n, :n], from_logits=True\n"
    "                )\n"
    "            )\n"
    "            c1, c2 = self.cluster_head(z1), self.cluster_head(z2)\n"
    "            loss_c = tf.reduce_mean(\n"
    "                tf.reduce_sum(c1*(1-c1), axis=1) + tf.reduce_sum(c2*(1-c2), axis=1)\n"
    "            )\n"
    "            # Pérdida de clasificación sobre datos etiquetados\n"
    "            z_lab  = self._encode_flat(x_lab, training=True)\n"
    "            pred   = self._cls_forward(z_lab, training=True)\n"
    "            loss_s = self._cce(y_lab, pred)\n"
    "\n"
    "            total = loss_m + self.lambda_ * loss_c + self.alpha * loss_s\n"
    "\n"
    "        all_vars = (self.encoder.trainable_variables\n"
    "                    + self.cluster_head.trainable_variables\n"
    "                    + [v for layer in self._cls_layers\n"
    "                       for v in layer.trainable_variables])\n"
    "        self.optimizer.apply_gradients(zip(tape.gradient(total, all_vars), all_vars))\n"
    "        return total, loss_m, loss_c, loss_s\n"
    "\n"
    "    def fit(self, x_all, x_labeled, y_labeled, x_val, y_val, epochs=8, batch_size=128):\n"
    "        \"\"\"\n"
    "        Entrenamiento conjunto contrastivo + clasificación.\n"
    "        Returns historial de pérdidas y accuracy de validación.\n"
    "        \"\"\"\n"
    "        # Inicializar pesos\n"
    "        _ = self.encoder(tf.zeros((1, 32, 32, 3)))\n"
    "        _ = self._cls_forward(tf.zeros((1, int(4*4*CFG[\"filters\"][-1]))))\n"
    "\n"
    "        ds_all = (tf.data.Dataset.from_tensor_slices(x_all)\n"
    "                  .shuffle(len(x_all), seed=SEED)\n"
    "                  .batch(batch_size, drop_remainder=True)\n"
    "                  .prefetch(tf.data.AUTOTUNE))\n"
    "        ds_lab = (tf.data.Dataset.from_tensor_slices((x_labeled, y_labeled))\n"
    "                  .shuffle(len(x_labeled), seed=SEED)\n"
    "                  .batch(batch_size)\n"
    "                  .repeat()          # repite para sincronizar con ds_all\n"
    "                  .prefetch(tf.data.AUTOTUNE))\n"
    "\n"
    "        history = {\"total_loss\": [], \"val_accuracy\": []}\n"
    "        for ep in range(epochs):\n"
    "            losses = []\n"
    "            for (x_b_all, (x_b_lab, y_b_lab)) in zip(ds_all, ds_lab):\n"
    "                tl, _, _, _ = self._train_step(x_b_all, x_b_lab, y_b_lab)\n"
    "                losses.append(float(tl))\n"
    "            # Validación\n"
    "            z_v   = self._encode_flat(x_val, training=False)\n"
    "            pred_v = self._cls_forward(z_v, training=False).numpy()\n"
    "            val_acc = float(np.mean(pred_v.argmax(1) == y_val.argmax(1)))\n"
    "            history[\"total_loss\"].append(np.mean(losses))\n"
    "            history[\"val_accuracy\"].append(val_acc)\n"
    "            print(f\"Época {ep+1}/{epochs}  loss={np.mean(losses):.4f}  \"\n"
    "                  f\"val_acc={val_acc:.4f}\")\n"
    "        return history\n"
    "\n"
    "    def evaluate(self, x, y):\n"
    "        z = self._encode_flat(x, training=False)\n"
    "        pred = self._cls_forward(z, training=False).numpy()\n"
    "        return float(self._cce(y, pred)), float(np.mean(pred.argmax(1) == y.argmax(1)))\n"
))

cells.append(code_cell(
    "joint_e6_4 = JointContrastiveClassifier(\n"
    "    num_classes=CFG[\"num_classes\"], num_clusters=CFG[\"num_classes\"],\n"
    "    filters=CFG[\"filters\"], dense_units=CFG[\"dense_units\"],\n"
    "    dropout=CFG[\"dropout\"], l2_reg=CFG[\"l2_reg\"],\n"
    "    tau=CFG[\"cl_tau\"], cl_lambda=CFG[\"cl_lambda\"],\n"
    "    alpha=CFG[\"cl_alpha\"], lr=CFG[\"lr\"],\n"
    ")\n"
    "\n"
    "history_e6_4 = joint_e6_4.fit(\n"
    "    x_all_e6, x_train, y_train, x_val, y_val,\n"
    "    epochs=CFG[\"cl_epochs\"], batch_size=CFG[\"ae_batch_size\"],\n"
    ")\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n"
    "ep = range(1, len(history_e6_4[\"total_loss\"]) + 1)\n"
    "ax1.plot(ep, history_e6_4[\"total_loss\"], \"o-\")\n"
    "ax1.set_title(\"E6.4 — Pérdida conjunta contrastiva\")\n"
    "ax1.set_xlabel(\"Época\")\n"
    "ax1.grid(linewidth=0.5)\n"
    "ax2.plot(ep, history_e6_4[\"val_accuracy\"], \"o-\", color=\"green\")\n"
    "ax2.set_title(\"E6.4 — Accuracy en validación\")\n"
    "ax2.set_xlabel(\"Época\")\n"
    "ax2.grid(linewidth=0.5)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "_, test_acc_e6_4 = joint_e6_4.evaluate(x_test, y_test)\n"
    "print(f\"[Test E6.4]  Accuracy: {test_acc_e6_4:.4f}\")\n"
    "RESULTS[\"E6.4 JointContrastivo\"] = {\"accuracy\": test_acc_e6_4}\n"
))

# E6.5
cells.append(md_cell("### Ejercicio 6.5 — Filtrado de anomalías + pre-entrenamiento contrastivo"))

cells.append(code_cell(
    "# Reutilizamos el detector entrenado en E5 y los datos filtrados x_unlabeled_clean\n"
    "enc_e6_5 = ConvEncoderSubclass(filters=CFG[\"filters\"], dropout=CFG[\"dropout\"])\n"
    "_ = enc_e6_5(tf.zeros((1, 32, 32, 3)))\n"
    "\n"
    "pt_e6_5  = ContrastivePretrainer(enc_e6_5, num_clusters=CFG[\"num_classes\"],\n"
    "                                  tau=CFG[\"cl_tau\"], lambda_=CFG[\"cl_lambda\"], lr=CFG[\"lr\"])\n"
    "x_all_e6_5 = np.concatenate([x_train, x_unlabeled_clean], axis=0)\n"
    "cl_losses_e6_5 = pt_e6_5.fit(x_all_e6_5, epochs=CFG[\"cl_epochs\"])\n"
    "\n"
    "plt.figure(figsize=(7, 4))\n"
    "plt.plot(range(1, len(cl_losses_e6_5)+1), cl_losses_e6_5, \"o-\")\n"
    "plt.title(\"E6.5 — Pérdida contrastiva (datos filtrados)\")\n"
    "plt.xlabel(\"Época\")\n"
    "plt.ylabel(\"Loss\")\n"
    "plt.grid(linewidth=0.5)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "z_train_65 = extract_features(enc_e6_5, x_train)\n"
    "z_val_65   = extract_features(enc_e6_5, x_val)\n"
    "z_test_65  = extract_features(enc_e6_5, x_test)\n"
    "\n"
    "clf_e6_5 = build_cl_classifier(z_train_65.shape[1], CFG[\"num_classes\"],\n"
    "                                CFG[\"dense_units\"], CFG[\"dropout\"], CFG[\"l2_reg\"])\n"
    "clf_e6_5.compile(optimizer=make_optimizer(), loss=\"categorical_crossentropy\",\n"
    "                 metrics=[\"accuracy\"])\n"
    "h_cls_e6_5 = clf_e6_5.fit(\n"
    "    z_train_65, y_train,\n"
    "    epochs=CFG[\"epochs_cls\"],\n"
    "    batch_size=CFG[\"batch_size\"],\n"
    "    validation_data=(z_val_65, y_val),\n"
    "    callbacks=[\n"
    "        EarlyStopping(monitor=\"val_accuracy\", patience=8, restore_best_weights=True),\n"
    "        ReduceLROnPlateau(monitor=\"val_loss\", factor=0.5, patience=4),\n"
    "    ],\n"
    "    verbose=1,\n"
    ")\n"
    "plot_training(h_cls_e6_5, title=\"E6.5 — Clasificador contrastivo (filtrado)\")\n"
    "RESULTS[\"E6.5 Anomaly+Contrastivo\"] = evaluate_and_report(\n"
    "    clf_e6_5, z_test_65, y_test, \"Test E6.5\"\n"
    ")\n"
))

cells.append(md_cell(
    "### Ejercicio 6 — Preguntas\n\n"
    "*(Responder las preguntas de los apartados 3, 4 y 5 para las variantes E6.3, E6.4, E6.5.)*"
))

# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md_cell("---\n## Resumen de resultados"))

cells.append(code_cell(
    "print(\"\\n\" + \"=\" * 55)\n"
    "print(f\"{'Método':<30} {'Test Accuracy':>12}\")\n"
    "print(\"=\" * 55)\n"
    "for name, metrics in RESULTS.items():\n"
    "    print(f\"{name:<30} {metrics.get('accuracy', float('nan')):>12.4f}\")\n"
    "print(\"=\" * 55)\n"
    "\n"
    "methods = list(RESULTS.keys())\n"
    "accs    = [RESULTS[m].get(\"accuracy\", 0) for m in methods]\n"
    "colors  = plt.cm.tab10(np.linspace(0, 1, len(methods)))\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(13, 5))\n"
    "bars = ax.bar(methods, accs, color=colors, edgecolor=\"white\", width=0.6)\n"
    "ax.bar_label(bars, fmt=\"%.4f\", padding=3, fontsize=9)\n"
    "ax.set_ylim(0, min(1.0, max(accs) * 1.15))\n"
    "ax.set_ylabel(\"Accuracy en Test\")\n"
    "ax.set_title(\"Comparación de métodos — CIFAR-100 semi-supervisado\")\n"
    "ax.tick_params(axis=\"x\", rotation=30)\n"
    "ax.grid(axis=\"y\", linewidth=0.5)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ══════════════════════════════════════════════════════════════════════════════
# ESCRIBIR NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = "P2_MAA2_profesional.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

with open(out, "r", encoding="utf-8") as f:
    nb = json.load(f)
print(f"Notebook guardado: {out}  ({len(nb['cells'])} celdas)")
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])[:70].replace("\n", " ")
    print(f"  [{i:02d}] {c['cell_type']:8s} | {src}")
