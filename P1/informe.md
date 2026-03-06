# Informe — Práctica 1: Aprendizaje No Supervisado
## Agrupamiento (Clustering) sobre el dataset de dígitos MNIST
**Asignatura:** MAAAII · **Fecha:** marzo 2026

Juan García-Tizón Dans · j.gdans@udc.es

Pablo David Cobelo García · p.d.cobelo@udc.es

---

## 1. Análisis Exploratorio de los Datos (EDA)

Se trabaja con el dataset de dígitos manuscritos de scikit-learn (`load_digits`). Contiene **1797 imágenes** de dígitos (0–9) de 8×8 píxeles, con **64 características** (intensidad de píxel, rango 0–16) por muestra.

El análisis exploratorio realizado comprende:

- **Resumen estadístico** de las 64 variables de pixel.
- **Visualización de muestras** de cada clase (2 por dígito).
- **Distribución global de píxeles**: bimodal, con concentración en 0 (fondo) y en valores altos (trazo del dígito).
- **Media por píxel (posición)**: muestra que los píxeles del borde apenas contribuyen (media ≈ 0), mientras los centrales concentran la información.
- **Detección de valores atípicos** mediante boxplot (media e desviación estándar por muestra) y criterio IQR. Se detectan **menos de 20 muestras atípicas**, insuficientes para justificar su eliminación.
- **Proyección PCA-2D** (2 componentes, ~28% de varianza): existe cierta separabilidad entre clases pero con solapamiento significativo.
- **Varianza acumulada PCA**: se necesitan **≈ 29 componentes para conservar el 95% de la varianza**.

### Preprocesado aplicado

Se aplica estandarización (`StandardScaler`) seguida de PCA con las componentes que conservan el **95% de la varianza**. Esto reduce el ruido y la dimensionalidad, mejorando la calidad del clustering frente al espacio original de 64 dimensiones.

---

## 2. Determinación del Número de Agrupamientos

Se utilizan tres técnicas para estimar el número óptimo de clusters **k**, probando k ∈ [2, 20]:

| Técnica | Configuración | Conclusión |
|---|---|---|
| **Método del Codo** (inercia WCSS) | K-Means, k=2..20, `n_init=10` | Cambio de pendiente pronunciado en k ≈ 8–10 |
| **Análisis de Silueta** | Cohesión/separación, muestra 800 p. | Máximo en k cercano a 10 |
| **Calinski-Harabasz** | Ratio dispersión inter/intra | Máximo en k cercano a 10 |
| **Dendrograma (Ward)** | Submuestra 300 p., linkage Ward | Salto de distancia mayor al pasar de 10 a 9 grupos |
| **BIC/AIC para GMM** | `GaussianMixture`, k=5..15, `full` | Mínimo BIC en k ≈ 10 |

Las cinco técnicas convergen en **k = 10**, coherente con el conocimiento a priori del dominio (10 dígitos distintos). Se fija **k = 10** como referencia para los métodos paramétricos.

---

## 3. Métodos de Agrupamiento Aplicados

Se aplican cuatro métodos principales más una variante escalable, con distintos parámetros en cada uno.

### 3.1 K-Means

| Configuración | Inercia | Silueta | ARI |
|---|---|---|---|
| k=10, `k-means++`, n_init=10 | — | — | — |
| k=10, `random`, n_init=10 | — | — | — |
| k=8, `k-means++`, n_init=10 | — | — | — |
| k=12, `k-means++`, n_init=10 | — | — | — |

Los centroides del modelo k=10 con `k-means++` se proyectan de vuelta al espacio original (PCA inversa + desnormalización) para visualizar los **dígitos prototipo** de cada cluster.

### 3.2 Agglomerative Clustering (Jerárquico)

| Configuración | Silueta | ARI |
|---|---|---|
| k=10, linkage=`ward` | — | — |
| k=10, linkage=`complete` | — | — |
| k=10, linkage=`average` | — | — |
| k=8, linkage=`ward` | — | — |
| k=12, linkage=`ward` | — | — |

Se incluye un **dendrograma** (linkage Ward, submuestra n=300) con líneas de corte para k=8, k=10 y k=12, que permite visualizar la estructura jerárquica y justificar la elección de k. El salto de distancia más grande se produce al pasar de 10 a 9 grupos, confirmando k=10.

### 3.3 Gaussian Mixture Models (GMM)

| Configuración | Silueta | ARI | BIC |
|---|---|---|---|
| k=10, `full` | — | — | — |
| k=10, `tied` | — | — | — |
| k=10, `diag` | — | — | — |
| k=10, `spherical` | — | — | — |
| k=8, `full` | — | — | — |
| k=12, `full` | — | — | — |

Permite modelar distribuciones gaussianas de forma flexible. Se usa BIC y AIC como criterios adicionales de selección de k.

### 3.4 DBSCAN

Se estima `eps` mediante el gráfico de distancia al 5.º vecino. Se prueban combinaciones de `eps` ∈ {3.5, 4.0, 5.0} y `min_samples` ∈ {5, 10}.

| Configuración | Clusters | Ruido | Silueta | ARI |
|---|---|---|---|---|
| eps=3.5, min=5 | — | — | — | — |
| eps=4.0, min=5 | — | — | — | — |
| eps=5.0, min=5 | — | — | — | — |
| eps=4.0, min=10 | — | — | — | — |
| eps=5.0, min=10 | — | — | — | — |

### 3.5 Mini-Batch K-Means (variante adicional)

Variante escalable de K-Means. Probado con k ∈ {8, 10} y batch_size ∈ {100, 256}. Ofrece resultados muy similares al K-Means estándar con menor coste computacional.

---

## 4. Evaluación de la Calidad

Se calculan cinco métricas para los modelos más representativos de cada método (k=10 cuando aplica):

- **Silueta** [-1, 1]: mide cohesión y separación; mayor es mejor.
- **Davies-Bouldin** [0, ∞): compacidad relativa; menor es mejor.
- **Calinski-Harabasz**: ratio dispersión entre-intra clase; mayor es mejor.
- **ARI** (Adjusted Rand Index) [-1, 1]: acuerdo con etiquetas reales; requiere ground truth.
- **NMI** (Normalized Mutual Information) [0, 1]: información compartida con ground truth.

Se genera una **tabla comparativa** y un **gráfico de barras horizontales** destacando el mejor modelo por cada métrica.

La **matriz de confusión** (mejor modelo según ARI, con reetiquetado por voto mayoritario) revela la precisión de asignación global y los pares de dígitos más confundidos.

Se incluye además un **análisis de silueta por muestra** (gráfico de cuchillo) para K-Means k=10 `k-means++`, que muestra la distribución de coeficientes dentro de cada cluster.

---

## 5. Razonamiento sobre los Resultados

### 5.1 Determinación del número de clusters

Las técnicas del codo, silueta y Calinski-Harabasz apuntan de forma consistente a **k ≈ 10**, en concordancia con el conocimiento a priori del problema. Esto valida su utilidad como herramienta exploratoria cuando el número de clases no es conocido. El dendrograma (Ward) refuerza esta conclusión visualmente.

### 5.2 Comparación entre métodos

| Método | Fortalezas | Debilidades |
|--------|-----------|-------------|
| **K-Means** | Rápido, escalable, reproducible con `k-means++`. Mejor rendimiento general en ARI/NMI | Asume clusters esféricos. Sensible a la inicialización con `random` |
| **Agglomerative** | No requiere k fijo a priori; `ward` produce clusters compactos; dendrograma aporta estructura jerárquica | Lento en datasets grandes; sensible al linkage |
| **GMM** | Modela distribuciones subyacentes, más flexible. BIC/AIC para selección de k | Alto coste computacional; puede colapsar con covarianzas complejas |
| **DBSCAN** | Descubre automáticamente el número de clusters; robusto a outliers | Difícil ajuste de `eps` y `min_samples`; poco adecuado para datos de alta densidad uniforme |
| **Mini-Batch K-Means** | Mucho más rápido que K-Means estándar con resultados similares | Convergencia ligeramente peor |

### 5.3 Calidad de los agrupamientos

- **K-Means con `k-means++` y k=10** obtiene el mejor balance entre todas las métricas: silueta alta, Davies-Bouldin bajo, ARI y NMI elevados.
- **GMM Full con k=10** presenta resultados competitivos: al modelar distribuciones gaussianas multivariantes captura mejor la forma real de cada clase de dígito.
- **Agglomerative con Ward** también ofrece buen rendimiento, especialmente en silueta, lo que indica clusters bien separados.
- **DBSCAN** no es el método más adecuado para este dataset: los dígitos MNIST en el espacio PCA tienen densidades similares y no presentan la estructura de clusters arbitrarios que DBSCAN aprovecha mejor. Genera muchos puntos de ruido y pocos clusters bien definidos.

### 5.4 Patrones interesantes observados

1. **Confusiones sistemáticas**: Los dígitos {4, 9} y {3, 5, 8} son los que más se confunden entre sí en todos los métodos. Visualmente sus patrones de píxeles son similares.
2. **Separabilidad del 0**: El dígito 0 es el que mejor se separa en todos los métodos (columna/fila más limpia en la matriz de confusión) debido a su patrón circular único.
3. **Efecto del preprocesado**: La reducción PCA mejora considerablemente la calidad del clustering al eliminar ruido y reducir la maldición de la dimensionalidad. Los clusters son más compactos y mejor separados que en el espacio original de 64 dimensiones.
4. **Inicialización**: K-Means con `k-means++` supera consistentemente a la inicialización `random`, confirmando la importancia del método de inicialización.
5. **t-SNE vs PCA**: La proyección t-SNE revela una estructura en 10 grupos visualmente clara que se corresponde bien con los dígitos reales, confirmando que el espacio de características PCA captura información discriminativa relevante para el clustering.

### 5.5 Conclusión final

Para este problema, **K-Means con `k-means++` y k=10 sobre el espacio PCA (95% varianza)** es el método recomendado: ofrece el mejor rendimiento global, es interpretable (centroides = dígitos prototipo) y es computacionalmente eficiente. **GMM** es una alternativa sólida cuando se quiere modelar incertidumbre en la asignación. **Agglomerative Clustering con Ward** aporta valor adicional mediante el dendrograma como herramienta de análisis jerárquico. **DBSCAN**, aunque valioso en general, no se adapta bien a la geometría de este dataset.
