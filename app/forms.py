import numpy as np
from django import forms

from . import settings
from .core.variable_type_checking import var_type

VAR_TYPES_LABEL, VAR_TYPES_CHOICES = settings.VAR_TYPES
PRE_SELECT = 3

CHECKBOXS = [
    settings.IMPUTE_METHODS,
    settings.SAMPLING_METHODS,
    settings.SELE_METHODS,
    settings.TRAIN_METHODS
]


class ConfigForm(forms.Form):

    def __init__(self, *args, **kwargs):
        variables = kwargs.pop('variables')
        data_df = kwargs.pop('data_df')
        super(ConfigForm, self).__init__(*args, **kwargs)
        self.fields['Target Column'] = forms.CharField(
            widget=forms.TextInput(attrs={
                'required': True,
            }),
        )
        self.initial['Target Column'] = variables[0]
        for i, variable in enumerate(variables):
            try:
                col = np.sort(data_df[variable].unique())
            except:
                col = data_df[variable].unique()
            print(variable, col)
            self.fields[f'var-{variable}'] = forms.ChoiceField(
                widget=forms.RadioSelect(attrs={
                    'required': 'true'
                }),
                label=f'{variable} {col[:3]}',
                choices=list(VAR_TYPES_CHOICES.items()),
                required=True,
            )
            self.initial[f'var-{variable}'] = var_type(col)

        for label, choices in CHECKBOXS:
            flag = label in ('Train Methods', 'Impute Methods')
            attrs = {}
            self.fields[label] = forms.MultipleChoiceField(
                widget=forms.CheckboxSelectMultiple(attrs=attrs),
                label=label,
                choices=list(choices.items()),
                required=flag,
            )
            self.initial[label] = list(choices.keys())[
                :PRE_SELECT if flag else 2]
