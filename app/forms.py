import numpy as np
from django import forms

from . import settings
from .core.variable_type_checking import var_type


VAR_TYPES_LABEL, VAR_TYPES_CHOICES = settings.VAR_TYPES

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
        for i, variable in enumerate(variables):
            col = data_df[variable].unique()
            self.fields[f'var-{variable}'] = forms.ChoiceField(
                widget=forms.RadioSelect(attrs={
                    'required': 'true'
                }),
                label=f'{variable} {np.random.choice(col, size=3)}',
                choices=list(VAR_TYPES_CHOICES.items()),
                required=True,
            )
            self.initial[f'var-{variable}'] = var_type(col)

        for label, choices in CHECKBOXS:
            attrs = {}
            self.fields[label] = forms.MultipleChoiceField(
                widget=forms.CheckboxSelectMultiple(attrs=attrs),
                label=label,
                choices=list(choices.items()),
                required=True,
            )
