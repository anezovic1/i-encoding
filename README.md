# i-encoding

## Authors

- [Anida Nezovic](https://github.com/anezovic1)
- [Dr Aida Brankovic](https://scholar.google.it/citations?user=Lh3kj1MAAAAJ&hl=en)
- [Dr Jin Yoon](https://scholar.google.com.au/citations?user=Ol5i7bcAAAAJ&hl=en)

## Overview

The <code>IEncoder</code> is a custom encoder designed to transform categorical variables into numerical representations using a unique encoding technique. The <code>fit_transform</code> method is a key part of its functionality, combining the fitting and transformation processes into a single step.

### Purpose

The method encodes categorical features into numerical values by mapping each category to a unique angular representation. This approach ensures a compact and continuous numerical representation of categorical variables while excluding a target column.

### How It Works

The method starts by validating the input data <code>X</code>. It checks the format, dimensionality and ensures no invalid values (like NaN or inf) are present. Using the <code>fit</code> method, it identifies the categorical features in the dataset.

Each category is mapped to a unique angle in radians using a circular mapping strategy (2π divided by the number of categories).

If the <code>target_column</code> parameter is specified, the transformed dataset excludes the target column, as it is not meant to be encoded.

The final transformed dataset is returned as a pandas DataFrame, preserving the original feature names.

## Requirements

The package depends on the following libraries:

- numpy
- pandas
- scikit-learn

## Installation

i-encoder is on PyPi and can be installed using pip:

```bash
pip install iencoder
```

## Contact

If you have any questions, suggestions or feedback, feel free to reach out:

- Email: nezovicanida@gmail.com

## License

This project is licensed under the [MIT License](LICENSE).
