from django.db import models


class Config(models.Model):
    id = models.AutoField(primary_key=True)
    data_name = models.CharField(max_length=100)
    tgt_col = models.CharField(max_length=100)
    variables = models.CharField(max_length=500)
    var_types = models.CharField(max_length=200)
    imp_method = models.CharField(max_length=100)
    sampling_method = models.CharField(max_length=100)
    sele_method = models.CharField(max_length=100)
    model_name = models.CharField(max_length=300)

    def __str__(self):
        return self.data_name
