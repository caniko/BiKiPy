=========
PolygonPerimeter
=========
The :code:`PolygonPerimeter` class and its daughters are central to every behavioural analysis workflow in BiKiPy. These classes are what defines different areas in an experimental setup. These perimeters are user defined, and can take any polygonal shape.

Most of the end-users will use the :code:`PolygonPerimeter` class. This class is versatile with respect to number of sides in our geometric shape. With that said, there are native classes for triangle, :code:`TriangularPerimeter`, and parallelogram, :code:`ParallelogramPerimeter`.

.. note::
    Use the :code:`PolygonPerimeter.init_polygon` when instantiating your polygon. This function will detect the number of sides in your dataset, and pass on the proper keyword arguments to the correct class.

:code:`PerimeterSet` combines several perimeters into one set, and should be used when working with sets of perimeters.

.. note::
    The :code:`PolygonPerimeter` class can't be used standalone. Use either :code:`ParallelogramPerimeter`, or :code:`PerimeterSet`.
