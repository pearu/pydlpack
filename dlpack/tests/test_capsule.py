from dlpack import Capsule


def test_new():
    o = object()
    c = Capsule.new(id(o), "mycapsule")
    assert c.get_name() == "mycapsule"
    assert c.is_valid("mycapsule")
    assert not c.is_valid("notmycapsule")
    assert c.get_pointer("mycapsule") == id(o)


def test_new_name_is_None():
    o = object()
    c = Capsule.new(id(o), None)  # name is None
    assert c.get_name() == None
    assert c.get_pointer(None) == id(o)


def test_set_name():
    o = object()
    c = Capsule.new(id(o), "mycapsule")
    assert c.get_name() == "mycapsule"
    assert c.is_valid("mycapsule")
    assert not c.is_valid("yourcapsule")
    c.set_name("yourcapsule")
    assert c.get_name() == "yourcapsule"
    assert not c.is_valid("mycapsule")
    assert c.is_valid("yourcapsule")


def test_set_name_is_None():
    o = object()
    c = Capsule.new(id(o), None)
    assert c.get_name() == None
    assert c.is_valid(None)
    assert not c.is_valid("mycapsule")
    c.set_name("mycapsule")
    assert c.get_name() == "mycapsule"
    assert not c.is_valid(None)
    assert c.is_valid("mycapsule")


def test_set_pointer():
    o = object()
    o2 = object()
    c = Capsule.new(id(o), None)
    assert c.is_valid(None)
    assert c.get_pointer(None) == id(o)
    c.set_pointer(id(o2))
    assert c.get_pointer(None) == id(o2)
    assert c.is_valid(None)


def test_set_context():
    o = object()
    c = Capsule.new(id(o), None)
    assert c.is_valid(None)
    assert c.get_context() is None
    o2 = object()
    c.set_context(id(o2))
    assert c.get_context() == id(o2)
    c.set_context(None)
    assert c.get_context() is None
    assert c.is_valid(None)


def test_destructor():
    o = object()
    destructor_is_called = []

    def destructor(capsule):
        name = capsule.get_name()
        assert name == "mycapsule"
        assert capsule.is_valid(name)
        pointer = capsule.get_pointer(name)
        assert id(o) == pointer
        destructor_is_called.append(True)

    c = Capsule.new(id(o), "mycapsule", destructor)

    del c

    assert len(destructor_is_called) == 1


def test_set_destructor():
    o = object()
    destructor_is_called = []

    def destructor(capsule):
        name = capsule.get_name()
        assert name == "mycapsule"
        assert capsule.is_valid(name)
        pointer = capsule.get_pointer(name)
        assert id(o) == pointer
        destructor_is_called.append(True)

    c = Capsule.new(id(o), "mycapsule")

    c.set_destructor(destructor)

    del c

    assert len(destructor_is_called) == 1
