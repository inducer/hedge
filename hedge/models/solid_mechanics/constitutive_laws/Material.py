class Material:
    """
    Protocol declarator for materials
    """

    def celerity(self, Fn, ndm):
        """
        Calculate critical wave speed
        """

    def stress(self, Fn, ndm):
        """
        Calculate the stress
        """

    def tangent_moduli(self, Fn, ndf, ndm):
        """
        Compute elastic tangent moduli
        """
