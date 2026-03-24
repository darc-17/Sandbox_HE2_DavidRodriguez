# Referencia: Lectura de archivos en Pandas

Guia rapida de como importar datos en distintos formatos usando `pandas`.

---

## 1. CSV

```python
# CSV basico
df = pd.read_csv("datos/archivo.csv")
```

### CSV con separador personalizado y encoding latino

```python
df = pd.read_csv("datos/archivo.csv", sep=";", encoding="latin-1")
```

### Desde URL (por ejemplo, GitHub)

```python
url = "https://raw.githubusercontent.com/usuario/repositorio/main/archivo.csv"
df = pd.read_csv(url)
```

### CSV comprimido

```python
df = pd.read_csv("datos/archivo.zip", compression="zip")
```

---

## 2. Excel

### Leer una hoja especifica

```python
df = pd.read_excel("datos/archivo.xlsx", sheet_name="Hoja1")
```

### Leer todas las hojas como diccionario

```python
hojas = pd.read_excel("datos/archivo.xlsx", sheet_name=None)
# hojas es un dict: {"Hoja1": df1, "Hoja2": df2, ...}
```

### Inspeccionar nombres de hojas antes de leer

```python
archivo = pd.ExcelFile("datos/archivo.xlsx")
print(archivo.sheet_names)
```

---

## 3. Archivos de texto delimitado

### TXT separado por tabulador

```python
df = pd.read_csv("datos/archivo.txt", sep="\t")
```

### Archivo separado por pipes (|)

```python
df = pd.read_csv("datos/archivo_pipe.txt", sep="|")
```

### Archivos con espacios o delimitador multiple

```python
df = pd.read_csv("datos/archivo.dat", delim_whitespace=True)
```

---

## 4. JSON

### JSON plano

```python
df = pd.read_json("datos/archivo.json")
```

### JSON anidado (normalizar estructura)

```python
import json
from pandas import json_normalize

with open("datos/archivo.json") as f:
    data = json.load(f)
df = json_normalize(data, sep="_")
```

---

## 5. Parquet

```python
df = pd.read_parquet("datos/archivo.parquet")
```

> **Nota:** Parquet es un formato columnar muy eficiente para datos grandes. Requiere el paquete `pyarrow` o `fastparquet`.

---

## 6. Formatos estadisticos

### Stata (.dta)

```python
df = pd.read_stata("datos/archivo.dta")
```

### SPSS (.sav)

```python
df = pd.read_spss("datos/archivo.sav")
```

### SAS (.sas7bdat)

```python
df = pd.read_sas("datos/archivo.sas7bdat")
```

---

## 7. Archivos comprimidos

### ZIP

```python
df = pd.read_csv("datos/archivo.zip", compression="zip")
```

### GZIP

```python
df = pd.read_csv("datos/archivo.csv.gz", compression="gzip")
```

### TAR.GZ

```python
df = pd.read_csv("datos/archivo.tar.gz", compression="gzip")
```

---

## Resumen rapido

| Formato | Funcion | Parametros clave |
|---|---|---|
| CSV | `pd.read_csv()` | `sep`, `encoding`, `compression` |
| Excel | `pd.read_excel()` | `sheet_name` |
| JSON | `pd.read_json()` | `orient` |
| Parquet | `pd.read_parquet()` | `engine` |
| Stata | `pd.read_stata()` | - |
| SPSS | `pd.read_spss()` | - |
| SAS | `pd.read_sas()` | `format` |
