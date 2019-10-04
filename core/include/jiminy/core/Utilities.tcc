#include <memory>


namespace jiminy
{
    // *************************** Math *****************************

    template<typename T>
    T min(T && t)
    {
        return std::forward<T>(t);
    }

    template<typename T0, typename T1, typename... Ts>
    typename std::common_type<T0, T1, Ts...>::type min(T0 && val1, T1 && val2, Ts &&... vs)
    {
        if (val2 < val1)
        {
            return min(val2, std::forward<Ts>(vs)...);
        }
        else
        {
            return min(val1, std::forward<Ts>(vs)...);
        }
    }

    // *********************** Miscellaneous **************************

    template<class F, class dF=std::decay_t<F> >
    auto not_f(F&& f){
        return [f=std::forward<F>(f)](auto&&...args) mutable
               ->decltype(!std::declval<std::result_of_t<dF&(decltype(args)...)>>()) // optional, adds sfinae
               {
                   return !f(decltype(args)(args)...);
               };
    }

	template<typename KeyType, typename ValueType>
	std::vector<ValueType> getMapValues(std::map<KeyType, ValueType> m)
	{
		std::vector<ValueType> v;
        v.reserve(m.size());
		std::transform(m.begin(),
                       m.end(),
                       std::back_inserter(v),
                       [](std::pair<KeyType const, ValueType> & pair) -> ValueType
                       {
                           return pair.second;
                       });
		return v;
	}

	template<typename typeOut, typename typeIn>
	std::vector<std::shared_ptr<typeOut> > staticCastSharedPtrVector(std::vector<std::shared_ptr<typeIn> > vIn)
    {
		std::vector<std::shared_ptr<typeOut> > vOut;
        vOut.reserve(vIn.size());
		std::transform(vIn.begin(),
                       vIn.end(),
                       std::back_inserter(vIn),
                       [](std::shared_ptr<typeIn> & e) -> std::shared_ptr<typeOut>
                       {
                           return std::static_pointer_cast<typeOut>(e);
                       });
		return vOut;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Swap two blocks of data in a vector.
    ///
    /// \details Given two uneven blocks in a vector v = (... v1 ... v2 ...), this function modifies
    ///          v to v = (... v2 ... v1 ...). v1 and v2 can be of arbitrary size.
    ///
    /// \pre firstBlockStart + firstBlockLength <= secondBlockStart
    /// \pre secondBlockStart + secondBlockLength <= vector.size()
    ///
    /// \param[in, out] vector Vector to modify.
    /// \param[in] firstBlockStart Start index of the first block.
    /// \param[in] firstBlockLength Length of the first block.
    /// \param[in] secondBlockStart Start index of the second block.
    /// \param[in] secondBlockLength Length of the second block.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    template<typename type>
    void swapVectorBlocks(Eigen::Matrix<type, Eigen::Dynamic, 1>       & vector,
                          uint32_t                               const & firstBlockStart,
                          uint32_t                               const & firstBlockLength,
                          uint32_t                               const & secondBlockStart,
                          uint32_t                               const & secondBlockLength)
    {
        // Extract both blocks.
        Eigen::Matrix< type, Eigen::Dynamic, 1 > firstBlock = vector.segment(firstBlockStart, firstBlockLength);
        Eigen::Matrix< type, Eigen::Dynamic, 1 > secondBlock = vector.segment(secondBlockStart, secondBlockLength);

        // Extract content between the blocks.
        uint32_t middleLength = secondBlockStart - (firstBlockStart + firstBlockLength);
        Eigen::Matrix< type, Eigen::Dynamic, 1 > middleBlock = vector.segment(firstBlockStart + firstBlockLength, middleLength);

        // Reorder vector.
        vector.segment(firstBlockStart, secondBlockLength) = secondBlock;
        vector.segment(firstBlockStart + secondBlockLength, middleLength) = middleBlock;
        vector.segment(firstBlockStart + secondBlockLength + middleLength, firstBlockLength) = firstBlock;
    }
}