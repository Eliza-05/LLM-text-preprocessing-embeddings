# Preprocesamiento de Texto y Embeddings para LLMs

Este repositorio explora los fundamentos del preprocesamiento de texto para Large Language Models (LLMs), implementando desde cero los componentes esenciales: tokenización, construcción de vocabularios, generación de embeddings y preparación de datos para entrenamiento.

---

## Contenido del Repositorio

| Archivo | Descripción |
|---------|-------------|
| `embeddings.ipynb` | Implementación completa del pipeline de preprocesamiento con explicaciones detalladas |
| `the-verdict.txt` | Texto de ejemplo utilizado para demostración (cuento de Edith Wharton) |
| `README.md` | Documentación y resumen de conceptos clave |

---

## Conceptos Clave Implementados

### 1. Tokenización: Conversión de Texto a Unidades Discretas

La tokenización transforma texto continuo en tokens individuales mediante expresiones regulares que separan palabras y puntuación.

**Por qué es crítico:**
- Las redes neuronales procesan números, no texto raw
- Establece el vocabulario base del modelo
- Debe ser determinístico para garantizar reproducibilidad

**Validaciones esenciales:**
- Cada signo de puntuación como token independiente
- Proceso consistente: misma entrada = mismos tokens
- Preservación de información semántica

**Impacto de errores:**
- Tokenización incorrecta expande vocabulario innecesariamente (50K → 200K+ tokens)
- Procesos no determinísticos impiden convergencia durante entrenamiento

---

### 2. Conversión a IDs: Mapeo Token → Índice Numérico

Cada token único se mapea a un ID entero que sirve como índice en la matriz de embeddings.

**Por qué es necesario:**
- Permite indexación eficiente en estructuras de datos
- Habilita operaciones de backpropagation
- Facilita lookup de embeddings: `embedding_matrix[token_id]`

**Relación con arquitectura:**
- IDs son índices para extracción de vectores de embeddings
- Mapeo biyectivo garantiza decodificación correcta
- Inconsistencias rompen compatibilidad entre modelo y datos

**Estrategia BPE:**
- Maneja palabras desconocidas mediante descomposición en subpalabras
- Vocabulario más compacto y robusto
- Generalización superior a vocabularios fijos

---

### 3. Sliding Window: Generación de Secuencias de Entrenamiento

Crea pares (input, target) donde el target está desplazado una posición, simulando predicción del siguiente token.

**Configuración:**
```
max_length = 4, stride = 2:
  Ventana 1: input=[0,1,2,3] → target=[1,2,3,4]
  Ventana 2: input=[2,3,4,5] → target=[3,4,5,6]
  Ventana 3: input=[4,5,6,7] → target=[5,6,7,8]
```

**Ventajas del overlap:**
- **stride = max_length**: Sin overlap, mínimos ejemplos
- **stride = max_length/2**: Balance óptimo entre datos y eficiencia
- **stride = 1**: Máximo overlap, máximos ejemplos

**Impacto en aprendizaje:**
- Overlap aumenta contextos en los que aparece cada token
- Modelo aprende transiciones token-a-token completas
- Reduce overfitting mediante data augmentation implícito

---

### 4. Embeddings: De Símbolos Discretos a Representaciones Continuas

#### ¿Qué son los embeddings?

Una matriz de pesos entrenable `[vocab_size × embedding_dim]` donde cada fila representa un token como vector denso.

#### Aprendizaje de significado

Los embeddings **no se programan**, emergen del entrenamiento:

1. **Inicialización:** Vectores aleatorios sin significado
2. **Entrenamiento:** Predicción de siguiente token ajusta vectores
3. **Convergencia:** Palabras en contextos similares → vectores cercanos

#### Relación con redes neuronales

| Concepto NN | Aplicación en Embeddings |
|-------------|--------------------------|
| Pesos entrenables | Matriz de embeddings actualizada por backpropagation |
| Reducción dimensional | [50K dims one-hot] → [256 dims densos] |
| Espacio latente | Distancia euclidiana = similitud semántica |
| Composicionalidad | token_emb + pos_emb = input_embedding |
| Gradientes | Flujo desde loss hasta embeddings |

#### Propiedades emergentes

```
cosine_similarity(cat, dog) ≈ 0.85
king - man + woman ≈ queen
```

**Para sistemas agénticos:**
- "deploy", "launch", "start" agrupados → inferencia robusta
- "first...then" codifica temporalidad → razonamiento secuencial

---

## Experimento: Impacto de `max_length` y `stride`

### Configuración

Texto de prueba: 5,145 tokens (the-verdict.txt)

### Resultados Obtenidos

| max_length | stride | Muestras Generadas | Tipo de Overlap |
|------------|--------|-------------------|-----------------|
| 4 | 4 | ~1,286 | Sin overlap |
| 4 | 2 | ~2,570 | 50% overlap |
| 4 | 1 | ~5,141 | Máximo overlap |
| 8 | 8 | ~643 | Sin overlap |
| 8 | 4 | ~1,286 | 50% overlap |

**Fórmula aproximada:**
```
num_samples ≈ (total_tokens - max_length) / stride
```

### Análisis de Trade-offs

| Aspecto | Sin Overlap | 50% Overlap | Máximo Overlap |
|---------|------------|-------------|----------------|
| Muestras de entrenamiento | Mínimo | Medio | Máximo |
| Tiempo de entrenamiento | Rápido | Medio | Lento |
| Calidad del modelo | Básica | Buena | Excelente |
| Uso de memoria | Bajo | Medio | Alto |
| Riesgo de overfitting | Alto | Medio | Bajo |

### Conclusiones del Experimento

1. **Overlap = Data Augmentation implícito**
   - Cada token se expone a múltiples contextos
   - Modelo aprende patrones lingüísticos más robustos

2. **Recomendación práctica: stride = max_length / 2**
   - Duplica muestras vs sin overlap
   - Costo computacional razonable
   - Balance óptimo calidad/eficiencia

3. **Impacto en dependencias lingüísticas**
   - Con overlap: "not only... but also" se captura completo
   - Sin overlap: frases pueden dividirse entre ventanas

4. **Validación empírica**
   - Overlap NO es desperdicio de recursos
   - Mejora fundamental en capacidad de generalización
   - Crítico para aprender patrones complejos del lenguaje

---

## Requisitos

```bash
pip install torch tiktoken
```

## Ejecución

```bash
jupyter notebook embeddings.ipynb
```

---
