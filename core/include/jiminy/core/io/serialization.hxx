#ifndef JIMINY_SERIALIZATION_HXX
#define JIMINY_SERIALIZATION_HXX

#include <any>  // `std::any`
#include <sstream>  // `std::istream`, `std::ostream`, `std::istringstream`, `std::ostringstream`, `std::streambuf`

#include <boost/archive/binary_oarchive_impl.hpp>
#include <boost/archive/binary_iarchive_impl.hpp>
#include <boost/archive/detail/register_archive.hpp>


// ********************************** stateful_binary_oarchive ********************************* //

namespace jiminy
{
    namespace archive
    {
        struct AnyState
        {
            std::any state_;
        };
    }

    using stateful_binary_oarchive_impl =
        boost::archive::binary_oarchive_impl<stateful_binary_oarchive,
                                             std::ostream::char_type,
                                             std::ostream::traits_type>;

    /// \brief Custom binary archive type to allow passing extra information when saving.
    ///
    /// \details This binary archive is not cross-platform at the time being. If this limitation
    ///          turns out to be a blocking issue, it can be easily fixed.
    ///          See official `boost::serialization` example:
    ///          https://github.com/boostorg/serialization/blob/develop/example/portable_binary_iarchive.hpp
    class stateful_binary_oarchive : public stateful_binary_oarchive_impl, public archive::AnyState
    {
        friend class boost::archive::detail::interface_oarchive<stateful_binary_oarchive>;
        friend class boost::archive::basic_binary_oarchive<stateful_binary_oarchive>;
        friend class boost::archive::basic_binary_oprimitive<stateful_binary_oarchive,
                                                             std::ostream::char_type,
                                                             std::ostream::traits_type>;
        friend class boost::archive::save_access;

    protected:
        template<class T>
        void save_override(T && t)
        {
            stateful_binary_oarchive_impl::save_override(t);
        }

    public:
        stateful_binary_oarchive(std::ostream & os, unsigned int flags = 0) :
        stateful_binary_oarchive_impl(os, flags)
        {
        }

        stateful_binary_oarchive(std::streambuf & bsb, unsigned int flags = 0) :
        stateful_binary_oarchive_impl(bsb, flags)
        {
        }
    };
}

BOOST_SERIALIZATION_REGISTER_ARCHIVE(jiminy::stateful_binary_oarchive)
// BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(jiminy::stateful_binary_oarchive)

// ********************************** stateful_binary_iarchive ********************************* //

namespace jiminy
{
    using stateful_binary_iarchive_impl =
        boost::archive::binary_iarchive_impl<stateful_binary_iarchive,
                                             std::istream::char_type,
                                             std::istream::traits_type>;

    /// \brief Custom binary archive type to allow passing extra information when loading.
    class stateful_binary_iarchive : public stateful_binary_iarchive_impl, public archive::AnyState
    {
        friend class boost::archive::detail::interface_iarchive<stateful_binary_iarchive>;
        friend class boost::archive::basic_binary_iarchive<stateful_binary_iarchive>;
        friend class boost::archive::basic_binary_iprimitive<stateful_binary_iarchive,
                                                             std::istream::char_type,
                                                             std::istream::traits_type>;
        friend class boost::archive::load_access;

    protected:
        template<class T>
        void load_override(T && t)
        {
            stateful_binary_iarchive_impl::load_override(t);
        }

    public:
        stateful_binary_iarchive(std::istream & is, unsigned int flags = 0) :
        stateful_binary_iarchive_impl(is, flags)
        {
        }

        stateful_binary_iarchive(std::streambuf & bsb, unsigned int flags = 0) :
        stateful_binary_iarchive_impl(bsb, flags)
        {
        }
    };
}

namespace Eigen::internal
{
    template<>
    struct traits<jiminy::stateful_binary_iarchive>
    {
        enum
        {
            Flags = 0
        };
    };
}

BOOST_SERIALIZATION_REGISTER_ARCHIVE(jiminy::stateful_binary_iarchive)
// BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(jiminy::stateful_binary_iarchive)

// ******************************** saveToBinary, loadFromBinary ******************************* //

namespace jiminy
{
    template<typename T>
    std::string saveToBinary(const T & obj)
    {
        std::ostringstream os;
        {
            stateful_binary_oarchive oa(os);
            oa << obj;
            return os.str();
        }
    }

    template<typename T>
    void loadFromBinary(T & obj, const std::string & data)
    {
        std::istringstream is(data);
        {
            stateful_binary_iarchive ia(is);
            ia >> obj;
        }
    }
}

#endif  // JIMINY_SERIALIZATION_HXX
