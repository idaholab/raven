"""Classes describing interface variables."""
from abc import ABC
from enum import Enum
from typing import Any, Optional
from xml.etree.ElementTree import Element, SubElement

from .enums import Fmi2Causality, Fmi2Initial, Fmi2Variability


class ScalarVariable(ABC):
    """Abstract FMI scalar variable definition.

    Args:
        name (str): Variable name
        causality (:obj:`Fmi2Causality`, optional): Variable causality
        description (str, optional): Variable description
        initial (:obj:`Fmi2Initial`, optional): Variable initial status
        variability (:obj:`Fmi2Variability`, optional): Variable variability
    """
    def __init__(
        self,
        name: str,
        causality: Optional[Fmi2Causality] = None,
        description: Optional[str] = None,
        initial: Optional[Fmi2Initial] = None,
        variability: Optional[Fmi2Variability] = None,
        getter: Any = None,
        setter: Any = None
    ):
        self.getter = getter
        self.setter = setter
        self.local_name = name.split(".")[-1]
        self.__attrs = {
            "name": name,
            "valueReference": None,
            "description": description,
            "causality": causality,
            "variability": variability,
            "initial": initial,
            # 'canHandleMultipleSetPerTimeInstant': # Only for ME
        }

    @property
    def causality(self) -> Optional[Fmi2Causality]:
        """:obj:`Fmi2Causality` or None: Variable causality - None if not set"""
        return self.__attrs["causality"]

    @property
    def description(self) -> Optional[str]:
        """str or None: Variable description - None if not set"""
        return self.__attrs["description"]

    @property
    def initial(self) -> Optional[Fmi2Initial]:
        """:obj:`Fmi2Initial` or None: Variable initial status - None if not set"""
        return self.__attrs["initial"]

    @property
    def name(self) -> str:
        """str: Variable name"""
        return self.__attrs["name"]

    @property
    def value_reference(self) -> int:
        """int: Variable reference index"""
        return self.__attrs["valueReference"]

    @value_reference.setter
    def value_reference(self, value: int):
        if self.__attrs["valueReference"] is not None:
            raise RuntimeError("Value reference already set.")
        self.__attrs["valueReference"] = value

    @property
    def variability(self) -> Optional[Fmi2Variability]:
        """:obj:`Fmi2Variability` or None: Variable variability - None if not set"""
        return self.__attrs["variability"]

    @staticmethod
    def requires_start(v: 'ScalarVariable') -> bool:
        """Test if a variable requires a start attribute

        Returns:
            True if successful, False otherwise
        """
        return (
            v.initial == Fmi2Initial.exact
            or v.initial == Fmi2Initial.approx
            or v.causality == Fmi2Causality.input
            or v.causality == Fmi2Causality.parameter
            or v.variability == Fmi2Variability.constant
        )

    def to_xml(self) -> Element:
        """Convert the variable to XML node.

        Returns
            xml.etree.ElementTree.Element: XML node
        """
        attrib = dict()
        for key, value in self.__attrs.items():
            if value is not None:
                attrib[key] = str(value.name if isinstance(value, Enum) else value)
        return Element("ScalarVariable", attrib)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}" \
               f"(name={self.name}, " \
               f"causality={self.causality}, " \
               f"variability={self.variability})"


class Real(ScalarVariable):
    def __init__(self, name: str, start: Optional[Any] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.__attrs = {"start": start}

    @property
    def start(self) -> Optional[Any]:
        return self.__attrs["start"]

    @start.setter
    def start(self, value: float):
        self.__attrs["start"] = value

    def to_xml(self) -> Element:
        attrib = dict()
        for key, value in self.__attrs.items():
            if value is not None:
                # In order to not loose precision, a number of this type should be 
                # stored on an XML file with at least 16 significant digits
                attrib[key] = f"{value:.16g}"
        parent = super().to_xml()
        SubElement(parent, "Real", attrib)

        return parent


class Integer(ScalarVariable):
    def __init__(self, name: str, start: Optional[Any] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.__attrs = {"start": start}

    @property
    def start(self) -> Optional[Any]:
        return self.__attrs["start"]

    @start.setter
    def start(self, value: float):
        self.__attrs["start"] = value

    def to_xml(self) -> Element:
        attrib = dict()
        for key, value in self.__attrs.items():
            if value is not None:
                attrib[key] = str(value)
        parent = super().to_xml()
        SubElement(parent, "Integer", attrib)

        return parent


class Boolean(ScalarVariable):
    def __init__(self, name: str, start: Optional[Any] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.__attrs = {"start": start}

    @property
    def start(self) -> Optional[Any]:
        return self.__attrs["start"]

    @start.setter
    def start(self, value: float):
        self.__attrs["start"] = value

    def to_xml(self) -> Element:
        attrib = dict()
        for key, value in self.__attrs.items():
            if value is not None:
                attrib[key] = str(value).lower()
        parent = super().to_xml()
        SubElement(parent, "Boolean", attrib)

        return parent


class String(ScalarVariable):
    def __init__(self, name: str, start: Optional[Any] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.__attrs = {"start": start}

    @property
    def start(self) -> Optional[Any]:
        return self.__attrs["start"]

    @start.setter
    def start(self, value: float):
        self.__attrs["start"] = value

    def to_xml(self) -> Element:
        attrib = dict()
        for key, value in self.__attrs.items():
            if value is not None:
                attrib[key] = str(value)
        parent = super().to_xml()
        SubElement(parent, "String", attrib)

        return parent
