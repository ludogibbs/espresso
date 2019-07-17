#
# Copyright (C) 2013-2019 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr, unique_ptr, weak_ptr
from libcpp cimport bool

from boost cimport string_ref

from .utils cimport Span
from .communication cimport MpiCallbacks

cdef extern from "script_interface/ObjectManager.hpp" namespace "ScriptInterface":
    cppclass ObjectManager:
        ObjectManager(MpiCallbacks *)

cdef extern from "script_interface/ScriptInterface.hpp" namespace "ScriptInterface":
    shared_ptr[ObjectManager] initialize(MpiCallbacks &)
    void initialize(ObjectManager *)
    cdef cppclass Variant:
        Variant()
        Variant(const Variant & )
        Variant & operator = (const Variant &)

    bool is_type[T](const Variant &)
    bool is_none(const Variant &)
    ctypedef unordered_map[string, Variant] VariantMap

cdef extern from "script_interface/get_value.hpp" namespace "ScriptInterface":
    T get_value[T](const Variant T)

cdef extern from "script_interface/ScriptInterface.hpp" namespace "ScriptInterface":
    Variant make_variant[T](const T & x)

    cdef cppclass ObjectHandle:
        const string name()
        VariantMap get_parameters() except +
        Span[const string_ref] valid_parameters() except +
        Variant get_parameter(const string & name) except +
        void set_parameter(const string & name, const Variant & value) except +
        Variant call_method(const string & name, const VariantMap & parameters) except +
        void set_state(map[string, Variant]) except +
        map[string, Variant] get_state() except +

cdef extern from "script_interface/ScriptInterface.hpp" namespace "ScriptInterface":
    cdef cppclass CreationPolicy:
        pass
    shared_ptr[ObjectHandle] make_shared(const string &, CreationPolicy, const VariantMap &) except +

cdef extern from "script_interface/ScriptInterface.hpp" namespace "ScriptInterface::CreationPolicy":
    CreationPolicy LOCAL
    CreationPolicy GLOBAL

cdef extern from "script_interface/ObjectManager.hpp" namespace "ScriptInterface":
    cppclass ObjectManager:
        shared_ptr[ObjectHandle] make_shared(const string &, CreationPolicy, const VariantMap &) except +
        string serialize(const shared_ptr[ObjectHandle] &) except +
        shared_ptr[ObjectHandle] unserialize(const string & state) except +

cdef void init(MpiCallbacks &)
