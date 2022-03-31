from __future__ import absolute_import, print_function


class State(object):
    """ overall state of an MDP """
    def __init__(self, name, attr_names, vals):
        """
        Initialize the state
        :param name: name of the state
        :param attr_names: attribute names comprising the state
        :param vals: values of each of the attributes comprising the state
        """
        self.__name = name
        self.__atrs = attr_names
        for i, attr in enumerate(attr_names):
            self.__setattr__(attr, vals[i])

    def __getitem__(self, item):
        """
        :param item: get an item using index
        :return: vals[item]
        """
        return self.__getattribute__(self.__atrs[item])

    def __len__(self):
        return len(self.__atrs)

    def __str__(self):
        return str([(k, self.__getattribute__(k)) for k in self.__atrs])

    def values(self):
        return [self.__getattribute__(k) for k in self.__atrs]

