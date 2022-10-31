# Genetic Programming

## About

[Genetic Programming](https://d1wqtxts1xzle7.cloudfront.net/5871345/10.1.1.61.352-with-cover-page-v2.pdf?Expires=1666798486&Signature=anFva5JTM5cWB79aGPDqTEqFe93pd7sYaf5G8I4QT89V~Bbd3DViA9bhNvLVVIBL8TgWMEUKva~9FYgtHR50HIsnUqiPTexjK2fbCDlDayyVoGBGr9F7gUvTBa9AQJQ3tADMZ0sxwoIx-xto4ilqB4IocCVwoCzr1mNVBtvYODbNjZBSdTzBIZWWXDN16rqfKNzFjSu89heM7K2S-4dAAmyW9Vy5qi1QCybD3lal7~6z1Bv40wiPxUjgt9duIhStVlgInF6PXQFhvJxlVB6AZqp8AUlqBMJqJ0rB4HU2nkSjcnLgWljPRYg0sBcPVBr3dC-EHbZR8p8-AGKQcQuDRA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) was introduced by JR Koza in 1992. Our GP algorithm is grammar-guided, and therefore it is a GGGP algorithm. Still, it can be used as a standard GP method, using grammars to define the terminal and function class.

## Parameters

The GP class requires the following Args:

* grammar (Grammar): The grammar used to guide the search.
* representation (Representation): The individual representation used by the GP program. The default is treebased_representation.
* problem (Problem): The problem we are solving. Either a SingleObjectiveProblem or a MultiObjectiveProblem.

::: geneticengine.algorithms.gp.gp.GP
