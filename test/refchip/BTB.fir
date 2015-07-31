circuit BTB :
  module BTB : 
    input clk : Clock
    input reset : UInt<1>
    input req : {valid : UInt<1>, bits : {addr : UInt<39>}}
    output resp : {valid : UInt<1>, bits : {taken : UInt<1>, mask : UInt<1>, bridx : UInt<1>, target : UInt<39>, entry : UInt<6>, bht : {history : UInt<7>, value : UInt<2>}}}
    input btb_update : {valid : UInt<1>, bits : {prediction : {valid : UInt<1>, bits : {taken : UInt<1>, mask : UInt<1>, bridx : UInt<1>, target : UInt<39>, entry : UInt<6>, bht : {history : UInt<7>, value : UInt<2>}}}, pc : UInt<39>, target : UInt<39>, taken : UInt<1>, isJump : UInt<1>, isReturn : UInt<1>, br_pc : UInt<39>}}
    input bht_update : {valid : UInt<1>, bits : {prediction : {valid : UInt<1>, bits : {taken : UInt<1>, mask : UInt<1>, bridx : UInt<1>, target : UInt<39>, entry : UInt<6>, bht : {history : UInt<7>, value : UInt<2>}}}, pc : UInt<39>, taken : UInt<1>, mispredict : UInt<1>}}
    input ras_update : {valid : UInt<1>, bits : {isCall : UInt<1>, isReturn : UInt<1>, returnAddr : UInt<39>, prediction : {valid : UInt<1>, bits : {taken : UInt<1>, mask : UInt<1>, bridx : UInt<1>, target : UInt<39>, entry : UInt<6>, bht : {history : UInt<7>, value : UInt<2>}}}}}
    input invalidate : UInt<1>
    
    reg idxValid : UInt<1>[62], clk, reset
    cmem idxs : UInt<12>[62], clk
    cmem idxPages : UInt<3>[62], clk
    cmem tgts : UInt<12>[62], clk
    cmem tgtPages : UInt<3>[62], clk
    cmem pages : UInt<27>[6], clk
    reg pageValid : UInt<6>, clk, reset
    onreset pageValid := UInt<6>(0)
    infer accessor T_180670 = idxPages[UInt<1>(0)]
    node T_180671 = dshl(UInt<1>(1), T_180670)
    node T_180672 = bits(T_180671, 5, 0)
    infer accessor T_180673 = idxPages[UInt<1>(1)]
    node T_180674 = dshl(UInt<1>(1), T_180673)
    node T_180675 = bits(T_180674, 5, 0)
    infer accessor T_180676 = idxPages[UInt<2>(2)]
    node T_180677 = dshl(UInt<1>(1), T_180676)
    node T_180678 = bits(T_180677, 5, 0)
    infer accessor T_180679 = idxPages[UInt<2>(3)]
    node T_180680 = dshl(UInt<1>(1), T_180679)
    node T_180681 = bits(T_180680, 5, 0)
    infer accessor T_180682 = idxPages[UInt<3>(4)]
    node T_180683 = dshl(UInt<1>(1), T_180682)
    node T_180684 = bits(T_180683, 5, 0)
    infer accessor T_180685 = idxPages[UInt<3>(5)]
    node T_180686 = dshl(UInt<1>(1), T_180685)
    node T_180687 = bits(T_180686, 5, 0)
    infer accessor T_180688 = idxPages[UInt<3>(6)]
    node T_180689 = dshl(UInt<1>(1), T_180688)
    node T_180690 = bits(T_180689, 5, 0)
    infer accessor T_180691 = idxPages[UInt<3>(7)]
    node T_180692 = dshl(UInt<1>(1), T_180691)
    node T_180693 = bits(T_180692, 5, 0)
    infer accessor T_180694 = idxPages[UInt<4>(8)]
    node T_180695 = dshl(UInt<1>(1), T_180694)
    node T_180696 = bits(T_180695, 5, 0)
    infer accessor T_180697 = idxPages[UInt<4>(9)]
    node T_180698 = dshl(UInt<1>(1), T_180697)
    node T_180699 = bits(T_180698, 5, 0)
    infer accessor T_180700 = idxPages[UInt<4>(10)]
    node T_180701 = dshl(UInt<1>(1), T_180700)
    node T_180702 = bits(T_180701, 5, 0)
    infer accessor T_180703 = idxPages[UInt<4>(11)]
    node T_180704 = dshl(UInt<1>(1), T_180703)
    node T_180705 = bits(T_180704, 5, 0)
    infer accessor T_180706 = idxPages[UInt<4>(12)]
    node T_180707 = dshl(UInt<1>(1), T_180706)
    node T_180708 = bits(T_180707, 5, 0)
    infer accessor T_180709 = idxPages[UInt<4>(13)]
    node T_180710 = dshl(UInt<1>(1), T_180709)
    node T_180711 = bits(T_180710, 5, 0)
    infer accessor T_180712 = idxPages[UInt<4>(14)]
    node T_180713 = dshl(UInt<1>(1), T_180712)
    node T_180714 = bits(T_180713, 5, 0)
    infer accessor T_180715 = idxPages[UInt<4>(15)]
    node T_180716 = dshl(UInt<1>(1), T_180715)
    node T_180717 = bits(T_180716, 5, 0)
    infer accessor T_180718 = idxPages[UInt<5>(16)]
    node T_180719 = dshl(UInt<1>(1), T_180718)
    node T_180720 = bits(T_180719, 5, 0)
    infer accessor T_180721 = idxPages[UInt<5>(17)]
    node T_180722 = dshl(UInt<1>(1), T_180721)
    node T_180723 = bits(T_180722, 5, 0)
    infer accessor T_180724 = idxPages[UInt<5>(18)]
    node T_180725 = dshl(UInt<1>(1), T_180724)
    node T_180726 = bits(T_180725, 5, 0)
    infer accessor T_180727 = idxPages[UInt<5>(19)]
    node T_180728 = dshl(UInt<1>(1), T_180727)
    node T_180729 = bits(T_180728, 5, 0)
    infer accessor T_180730 = idxPages[UInt<5>(20)]
    node T_180731 = dshl(UInt<1>(1), T_180730)
    node T_180732 = bits(T_180731, 5, 0)
    infer accessor T_180733 = idxPages[UInt<5>(21)]
    node T_180734 = dshl(UInt<1>(1), T_180733)
    node T_180735 = bits(T_180734, 5, 0)
    infer accessor T_180736 = idxPages[UInt<5>(22)]
    node T_180737 = dshl(UInt<1>(1), T_180736)
    node T_180738 = bits(T_180737, 5, 0)
    infer accessor T_180739 = idxPages[UInt<5>(23)]
    node T_180740 = dshl(UInt<1>(1), T_180739)
    node T_180741 = bits(T_180740, 5, 0)
    infer accessor T_180742 = idxPages[UInt<5>(24)]
    node T_180743 = dshl(UInt<1>(1), T_180742)
    node T_180744 = bits(T_180743, 5, 0)
    infer accessor T_180745 = idxPages[UInt<5>(25)]
    node T_180746 = dshl(UInt<1>(1), T_180745)
    node T_180747 = bits(T_180746, 5, 0)
    infer accessor T_180748 = idxPages[UInt<5>(26)]
    node T_180749 = dshl(UInt<1>(1), T_180748)
    node T_180750 = bits(T_180749, 5, 0)
    infer accessor T_180751 = idxPages[UInt<5>(27)]
    node T_180752 = dshl(UInt<1>(1), T_180751)
    node T_180753 = bits(T_180752, 5, 0)
    infer accessor T_180754 = idxPages[UInt<5>(28)]
    node T_180755 = dshl(UInt<1>(1), T_180754)
    node T_180756 = bits(T_180755, 5, 0)
    infer accessor T_180757 = idxPages[UInt<5>(29)]
    node T_180758 = dshl(UInt<1>(1), T_180757)
    node T_180759 = bits(T_180758, 5, 0)
    infer accessor T_180760 = idxPages[UInt<5>(30)]
    node T_180761 = dshl(UInt<1>(1), T_180760)
    node T_180762 = bits(T_180761, 5, 0)
    infer accessor T_180763 = idxPages[UInt<5>(31)]
    node T_180764 = dshl(UInt<1>(1), T_180763)
    node T_180765 = bits(T_180764, 5, 0)
    infer accessor T_180766 = idxPages[UInt<6>(32)]
    node T_180767 = dshl(UInt<1>(1), T_180766)
    node T_180768 = bits(T_180767, 5, 0)
    infer accessor T_180769 = idxPages[UInt<6>(33)]
    node T_180770 = dshl(UInt<1>(1), T_180769)
    node T_180771 = bits(T_180770, 5, 0)
    infer accessor T_180772 = idxPages[UInt<6>(34)]
    node T_180773 = dshl(UInt<1>(1), T_180772)
    node T_180774 = bits(T_180773, 5, 0)
    infer accessor T_180775 = idxPages[UInt<6>(35)]
    node T_180776 = dshl(UInt<1>(1), T_180775)
    node T_180777 = bits(T_180776, 5, 0)
    infer accessor T_180778 = idxPages[UInt<6>(36)]
    node T_180779 = dshl(UInt<1>(1), T_180778)
    node T_180780 = bits(T_180779, 5, 0)
    infer accessor T_180781 = idxPages[UInt<6>(37)]
    node T_180782 = dshl(UInt<1>(1), T_180781)
    node T_180783 = bits(T_180782, 5, 0)
    infer accessor T_180784 = idxPages[UInt<6>(38)]
    node T_180785 = dshl(UInt<1>(1), T_180784)
    node T_180786 = bits(T_180785, 5, 0)
    infer accessor T_180787 = idxPages[UInt<6>(39)]
    node T_180788 = dshl(UInt<1>(1), T_180787)
    node T_180789 = bits(T_180788, 5, 0)
    infer accessor T_180790 = idxPages[UInt<6>(40)]
    node T_180791 = dshl(UInt<1>(1), T_180790)
    node T_180792 = bits(T_180791, 5, 0)
    infer accessor T_180793 = idxPages[UInt<6>(41)]
    node T_180794 = dshl(UInt<1>(1), T_180793)
    node T_180795 = bits(T_180794, 5, 0)
    infer accessor T_180796 = idxPages[UInt<6>(42)]
    node T_180797 = dshl(UInt<1>(1), T_180796)
    node T_180798 = bits(T_180797, 5, 0)
    infer accessor T_180799 = idxPages[UInt<6>(43)]
    node T_180800 = dshl(UInt<1>(1), T_180799)
    node T_180801 = bits(T_180800, 5, 0)
    infer accessor T_180802 = idxPages[UInt<6>(44)]
    node T_180803 = dshl(UInt<1>(1), T_180802)
    node T_180804 = bits(T_180803, 5, 0)
    infer accessor T_180805 = idxPages[UInt<6>(45)]
    node T_180806 = dshl(UInt<1>(1), T_180805)
    node T_180807 = bits(T_180806, 5, 0)
    infer accessor T_180808 = idxPages[UInt<6>(46)]
    node T_180809 = dshl(UInt<1>(1), T_180808)
    node T_180810 = bits(T_180809, 5, 0)
    infer accessor T_180811 = idxPages[UInt<6>(47)]
    node T_180812 = dshl(UInt<1>(1), T_180811)
    node T_180813 = bits(T_180812, 5, 0)
    infer accessor T_180814 = idxPages[UInt<6>(48)]
    node T_180815 = dshl(UInt<1>(1), T_180814)
    node T_180816 = bits(T_180815, 5, 0)
    infer accessor T_180817 = idxPages[UInt<6>(49)]
    node T_180818 = dshl(UInt<1>(1), T_180817)
    node T_180819 = bits(T_180818, 5, 0)
    infer accessor T_180820 = idxPages[UInt<6>(50)]
    node T_180821 = dshl(UInt<1>(1), T_180820)
    node T_180822 = bits(T_180821, 5, 0)
    infer accessor T_180823 = idxPages[UInt<6>(51)]
    node T_180824 = dshl(UInt<1>(1), T_180823)
    node T_180825 = bits(T_180824, 5, 0)
    infer accessor T_180826 = idxPages[UInt<6>(52)]
    node T_180827 = dshl(UInt<1>(1), T_180826)
    node T_180828 = bits(T_180827, 5, 0)
    infer accessor T_180829 = idxPages[UInt<6>(53)]
    node T_180830 = dshl(UInt<1>(1), T_180829)
    node T_180831 = bits(T_180830, 5, 0)
    infer accessor T_180832 = idxPages[UInt<6>(54)]
    node T_180833 = dshl(UInt<1>(1), T_180832)
    node T_180834 = bits(T_180833, 5, 0)
    infer accessor T_180835 = idxPages[UInt<6>(55)]
    node T_180836 = dshl(UInt<1>(1), T_180835)
    node T_180837 = bits(T_180836, 5, 0)
    infer accessor T_180838 = idxPages[UInt<6>(56)]
    node T_180839 = dshl(UInt<1>(1), T_180838)
    node T_180840 = bits(T_180839, 5, 0)
    infer accessor T_180841 = idxPages[UInt<6>(57)]
    node T_180842 = dshl(UInt<1>(1), T_180841)
    node T_180843 = bits(T_180842, 5, 0)
    infer accessor T_180844 = idxPages[UInt<6>(58)]
    node T_180845 = dshl(UInt<1>(1), T_180844)
    node T_180846 = bits(T_180845, 5, 0)
    infer accessor T_180847 = idxPages[UInt<6>(59)]
    node T_180848 = dshl(UInt<1>(1), T_180847)
    node T_180849 = bits(T_180848, 5, 0)
    infer accessor T_180850 = idxPages[UInt<6>(60)]
    node T_180851 = dshl(UInt<1>(1), T_180850)
    node T_180852 = bits(T_180851, 5, 0)
    infer accessor T_180853 = idxPages[UInt<6>(61)]
    node T_180854 = dshl(UInt<1>(1), T_180853)
    node T_180855 = bits(T_180854, 5, 0)
    infer accessor T_180856 = tgtPages[UInt<1>(0)]
    node T_180857 = dshl(UInt<1>(1), T_180856)
    node T_180858 = bits(T_180857, 5, 0)
    infer accessor T_180859 = tgtPages[UInt<1>(1)]
    node T_180860 = dshl(UInt<1>(1), T_180859)
    node T_180861 = bits(T_180860, 5, 0)
    infer accessor T_180862 = tgtPages[UInt<2>(2)]
    node T_180863 = dshl(UInt<1>(1), T_180862)
    node T_180864 = bits(T_180863, 5, 0)
    infer accessor T_180865 = tgtPages[UInt<2>(3)]
    node T_180866 = dshl(UInt<1>(1), T_180865)
    node T_180867 = bits(T_180866, 5, 0)
    infer accessor T_180868 = tgtPages[UInt<3>(4)]
    node T_180869 = dshl(UInt<1>(1), T_180868)
    node T_180870 = bits(T_180869, 5, 0)
    infer accessor T_180871 = tgtPages[UInt<3>(5)]
    node T_180872 = dshl(UInt<1>(1), T_180871)
    node T_180873 = bits(T_180872, 5, 0)
    infer accessor T_180874 = tgtPages[UInt<3>(6)]
    node T_180875 = dshl(UInt<1>(1), T_180874)
    node T_180876 = bits(T_180875, 5, 0)
    infer accessor T_180877 = tgtPages[UInt<3>(7)]
    node T_180878 = dshl(UInt<1>(1), T_180877)
    node T_180879 = bits(T_180878, 5, 0)
    infer accessor T_180880 = tgtPages[UInt<4>(8)]
    node T_180881 = dshl(UInt<1>(1), T_180880)
    node T_180882 = bits(T_180881, 5, 0)
    infer accessor T_180883 = tgtPages[UInt<4>(9)]
    node T_180884 = dshl(UInt<1>(1), T_180883)
    node T_180885 = bits(T_180884, 5, 0)
    infer accessor T_180886 = tgtPages[UInt<4>(10)]
    node T_180887 = dshl(UInt<1>(1), T_180886)
    node T_180888 = bits(T_180887, 5, 0)
    infer accessor T_180889 = tgtPages[UInt<4>(11)]
    node T_180890 = dshl(UInt<1>(1), T_180889)
    node T_180891 = bits(T_180890, 5, 0)
    infer accessor T_180892 = tgtPages[UInt<4>(12)]
    node T_180893 = dshl(UInt<1>(1), T_180892)
    node T_180894 = bits(T_180893, 5, 0)
    infer accessor T_180895 = tgtPages[UInt<4>(13)]
    node T_180896 = dshl(UInt<1>(1), T_180895)
    node T_180897 = bits(T_180896, 5, 0)
    infer accessor T_180898 = tgtPages[UInt<4>(14)]
    node T_180899 = dshl(UInt<1>(1), T_180898)
    node T_180900 = bits(T_180899, 5, 0)
    infer accessor T_180901 = tgtPages[UInt<4>(15)]
    node T_180902 = dshl(UInt<1>(1), T_180901)
    node T_180903 = bits(T_180902, 5, 0)
    infer accessor T_180904 = tgtPages[UInt<5>(16)]
    node T_180905 = dshl(UInt<1>(1), T_180904)
    node T_180906 = bits(T_180905, 5, 0)
    infer accessor T_180907 = tgtPages[UInt<5>(17)]
    node T_180908 = dshl(UInt<1>(1), T_180907)
    node T_180909 = bits(T_180908, 5, 0)
    infer accessor T_180910 = tgtPages[UInt<5>(18)]
    node T_180911 = dshl(UInt<1>(1), T_180910)
    node T_180912 = bits(T_180911, 5, 0)
    infer accessor T_180913 = tgtPages[UInt<5>(19)]
    node T_180914 = dshl(UInt<1>(1), T_180913)
    node T_180915 = bits(T_180914, 5, 0)
    infer accessor T_180916 = tgtPages[UInt<5>(20)]
    node T_180917 = dshl(UInt<1>(1), T_180916)
    node T_180918 = bits(T_180917, 5, 0)
    infer accessor T_180919 = tgtPages[UInt<5>(21)]
    node T_180920 = dshl(UInt<1>(1), T_180919)
    node T_180921 = bits(T_180920, 5, 0)
    infer accessor T_180922 = tgtPages[UInt<5>(22)]
    node T_180923 = dshl(UInt<1>(1), T_180922)
    node T_180924 = bits(T_180923, 5, 0)
    infer accessor T_180925 = tgtPages[UInt<5>(23)]
    node T_180926 = dshl(UInt<1>(1), T_180925)
    node T_180927 = bits(T_180926, 5, 0)
    infer accessor T_180928 = tgtPages[UInt<5>(24)]
    node T_180929 = dshl(UInt<1>(1), T_180928)
    node T_180930 = bits(T_180929, 5, 0)
    infer accessor T_180931 = tgtPages[UInt<5>(25)]
    node T_180932 = dshl(UInt<1>(1), T_180931)
    node T_180933 = bits(T_180932, 5, 0)
    infer accessor T_180934 = tgtPages[UInt<5>(26)]
    node T_180935 = dshl(UInt<1>(1), T_180934)
    node T_180936 = bits(T_180935, 5, 0)
    infer accessor T_180937 = tgtPages[UInt<5>(27)]
    node T_180938 = dshl(UInt<1>(1), T_180937)
    node T_180939 = bits(T_180938, 5, 0)
    infer accessor T_180940 = tgtPages[UInt<5>(28)]
    node T_180941 = dshl(UInt<1>(1), T_180940)
    node T_180942 = bits(T_180941, 5, 0)
    infer accessor T_180943 = tgtPages[UInt<5>(29)]
    node T_180944 = dshl(UInt<1>(1), T_180943)
    node T_180945 = bits(T_180944, 5, 0)
    infer accessor T_180946 = tgtPages[UInt<5>(30)]
    node T_180947 = dshl(UInt<1>(1), T_180946)
    node T_180948 = bits(T_180947, 5, 0)
    infer accessor T_180949 = tgtPages[UInt<5>(31)]
    node T_180950 = dshl(UInt<1>(1), T_180949)
    node T_180951 = bits(T_180950, 5, 0)
    infer accessor T_180952 = tgtPages[UInt<6>(32)]
    node T_180953 = dshl(UInt<1>(1), T_180952)
    node T_180954 = bits(T_180953, 5, 0)
    infer accessor T_180955 = tgtPages[UInt<6>(33)]
    node T_180956 = dshl(UInt<1>(1), T_180955)
    node T_180957 = bits(T_180956, 5, 0)
    infer accessor T_180958 = tgtPages[UInt<6>(34)]
    node T_180959 = dshl(UInt<1>(1), T_180958)
    node T_180960 = bits(T_180959, 5, 0)
    infer accessor T_180961 = tgtPages[UInt<6>(35)]
    node T_180962 = dshl(UInt<1>(1), T_180961)
    node T_180963 = bits(T_180962, 5, 0)
    infer accessor T_180964 = tgtPages[UInt<6>(36)]
    node T_180965 = dshl(UInt<1>(1), T_180964)
    node T_180966 = bits(T_180965, 5, 0)
    infer accessor T_180967 = tgtPages[UInt<6>(37)]
    node T_180968 = dshl(UInt<1>(1), T_180967)
    node T_180969 = bits(T_180968, 5, 0)
    infer accessor T_180970 = tgtPages[UInt<6>(38)]
    node T_180971 = dshl(UInt<1>(1), T_180970)
    node T_180972 = bits(T_180971, 5, 0)
    infer accessor T_180973 = tgtPages[UInt<6>(39)]
    node T_180974 = dshl(UInt<1>(1), T_180973)
    node T_180975 = bits(T_180974, 5, 0)
    infer accessor T_180976 = tgtPages[UInt<6>(40)]
    node T_180977 = dshl(UInt<1>(1), T_180976)
    node T_180978 = bits(T_180977, 5, 0)
    infer accessor T_180979 = tgtPages[UInt<6>(41)]
    node T_180980 = dshl(UInt<1>(1), T_180979)
    node T_180981 = bits(T_180980, 5, 0)
    infer accessor T_180982 = tgtPages[UInt<6>(42)]
    node T_180983 = dshl(UInt<1>(1), T_180982)
    node T_180984 = bits(T_180983, 5, 0)
    infer accessor T_180985 = tgtPages[UInt<6>(43)]
    node T_180986 = dshl(UInt<1>(1), T_180985)
    node T_180987 = bits(T_180986, 5, 0)
    infer accessor T_180988 = tgtPages[UInt<6>(44)]
    node T_180989 = dshl(UInt<1>(1), T_180988)
    node T_180990 = bits(T_180989, 5, 0)
    infer accessor T_180991 = tgtPages[UInt<6>(45)]
    node T_180992 = dshl(UInt<1>(1), T_180991)
    node T_180993 = bits(T_180992, 5, 0)
    infer accessor T_180994 = tgtPages[UInt<6>(46)]
    node T_180995 = dshl(UInt<1>(1), T_180994)
    node T_180996 = bits(T_180995, 5, 0)
    infer accessor T_180997 = tgtPages[UInt<6>(47)]
    node T_180998 = dshl(UInt<1>(1), T_180997)
    node T_180999 = bits(T_180998, 5, 0)
    infer accessor T_181000 = tgtPages[UInt<6>(48)]
    node T_181001 = dshl(UInt<1>(1), T_181000)
    node T_181002 = bits(T_181001, 5, 0)
    infer accessor T_181003 = tgtPages[UInt<6>(49)]
    node T_181004 = dshl(UInt<1>(1), T_181003)
    node T_181005 = bits(T_181004, 5, 0)
    infer accessor T_181006 = tgtPages[UInt<6>(50)]
    node T_181007 = dshl(UInt<1>(1), T_181006)
    node T_181008 = bits(T_181007, 5, 0)
    infer accessor T_181009 = tgtPages[UInt<6>(51)]
    node T_181010 = dshl(UInt<1>(1), T_181009)
    node T_181011 = bits(T_181010, 5, 0)
    infer accessor T_181012 = tgtPages[UInt<6>(52)]
    node T_181013 = dshl(UInt<1>(1), T_181012)
    node T_181014 = bits(T_181013, 5, 0)
    infer accessor T_181015 = tgtPages[UInt<6>(53)]
    node T_181016 = dshl(UInt<1>(1), T_181015)
    node T_181017 = bits(T_181016, 5, 0)
    infer accessor T_181018 = tgtPages[UInt<6>(54)]
    node T_181019 = dshl(UInt<1>(1), T_181018)
    node T_181020 = bits(T_181019, 5, 0)
    infer accessor T_181021 = tgtPages[UInt<6>(55)]
    node T_181022 = dshl(UInt<1>(1), T_181021)
    node T_181023 = bits(T_181022, 5, 0)
    infer accessor T_181024 = tgtPages[UInt<6>(56)]
    node T_181025 = dshl(UInt<1>(1), T_181024)
    node T_181026 = bits(T_181025, 5, 0)
    infer accessor T_181027 = tgtPages[UInt<6>(57)]
    node T_181028 = dshl(UInt<1>(1), T_181027)
    node T_181029 = bits(T_181028, 5, 0)
    infer accessor T_181030 = tgtPages[UInt<6>(58)]
    node T_181031 = dshl(UInt<1>(1), T_181030)
    node T_181032 = bits(T_181031, 5, 0)
    infer accessor T_181033 = tgtPages[UInt<6>(59)]
    node T_181034 = dshl(UInt<1>(1), T_181033)
    node T_181035 = bits(T_181034, 5, 0)
    infer accessor T_181036 = tgtPages[UInt<6>(60)]
    node T_181037 = dshl(UInt<1>(1), T_181036)
    node T_181038 = bits(T_181037, 5, 0)
    infer accessor T_181039 = tgtPages[UInt<6>(61)]
    node T_181040 = dshl(UInt<1>(1), T_181039)
    node T_181041 = bits(T_181040, 5, 0)
    reg useRAS : UInt<1>[62], clk, reset
    reg isJump : UInt<1>[62], clk, reset
    cmem brIdx : UInt<1>[62], clk
    reg T_181042 : UInt<1>, clk, reset
    onreset T_181042 := UInt<1>(0)
    T_181042 := btb_update.valid
    reg T_181043 : {prediction : {valid : UInt<1>, bits : {taken : UInt<1>, mask : UInt<1>, bridx : UInt<1>, target : UInt<39>, entry : UInt<6>, bht : {history : UInt<7>, value : UInt<2>}}}, pc : UInt<39>, target : UInt<39>, taken : UInt<1>, isJump : UInt<1>, isReturn : UInt<1>, br_pc : UInt<39>}, clk, reset
    when btb_update.valid : T_181043 := btb_update.bits
    r_btb_update.valid <> T_181042
    r_btb_update.bits <> T_181043
    node T_181044 = shr(req.bits.addr, 12)
    infer accessor T_181045 = pages[UInt<1>(0)]
    node T_181046 = eq(T_181045, T_181044)
    infer accessor T_181047 = pages[UInt<1>(1)]
    node T_181048 = eq(T_181047, T_181044)
    infer accessor T_181049 = pages[UInt<2>(2)]
    node T_181050 = eq(T_181049, T_181044)
    infer accessor T_181051 = pages[UInt<2>(3)]
    node T_181052 = eq(T_181051, T_181044)
    infer accessor T_181053 = pages[UInt<3>(4)]
    node T_181054 = eq(T_181053, T_181044)
    infer accessor T_181055 = pages[UInt<3>(5)]
    node T_181056 = eq(T_181055, T_181044)
    wire T_181057 : UInt<1>[6]
    T_181057[0] := T_181046
    T_181057[1] := T_181048
    T_181057[2] := T_181050
    T_181057[3] := T_181052
    T_181057[4] := T_181054
    T_181057[5] := T_181056
    node T_181058 = cat(T_181057[4], T_181057[3])
    node T_181059 = cat(T_181057[5], T_181058)
    node T_181060 = cat(T_181057[1], T_181057[0])
    node T_181061 = cat(T_181057[2], T_181060)
    node T_181062 = cat(T_181059, T_181061)
    node pageHit = and(T_181062, pageValid)
    node T_181063 = bits(req.bits.addr, 11, 0)
    infer accessor T_181064 = idxs[UInt<1>(0)]
    node T_181065 = eq(T_181064, T_181063)
    infer accessor T_181066 = idxs[UInt<1>(1)]
    node T_181067 = eq(T_181066, T_181063)
    infer accessor T_181068 = idxs[UInt<2>(2)]
    node T_181069 = eq(T_181068, T_181063)
    infer accessor T_181070 = idxs[UInt<2>(3)]
    node T_181071 = eq(T_181070, T_181063)
    infer accessor T_181072 = idxs[UInt<3>(4)]
    node T_181073 = eq(T_181072, T_181063)
    infer accessor T_181074 = idxs[UInt<3>(5)]
    node T_181075 = eq(T_181074, T_181063)
    infer accessor T_181076 = idxs[UInt<3>(6)]
    node T_181077 = eq(T_181076, T_181063)
    infer accessor T_181078 = idxs[UInt<3>(7)]
    node T_181079 = eq(T_181078, T_181063)
    infer accessor T_181080 = idxs[UInt<4>(8)]
    node T_181081 = eq(T_181080, T_181063)
    infer accessor T_181082 = idxs[UInt<4>(9)]
    node T_181083 = eq(T_181082, T_181063)
    infer accessor T_181084 = idxs[UInt<4>(10)]
    node T_181085 = eq(T_181084, T_181063)
    infer accessor T_181086 = idxs[UInt<4>(11)]
    node T_181087 = eq(T_181086, T_181063)
    infer accessor T_181088 = idxs[UInt<4>(12)]
    node T_181089 = eq(T_181088, T_181063)
    infer accessor T_181090 = idxs[UInt<4>(13)]
    node T_181091 = eq(T_181090, T_181063)
    infer accessor T_181092 = idxs[UInt<4>(14)]
    node T_181093 = eq(T_181092, T_181063)
    infer accessor T_181094 = idxs[UInt<4>(15)]
    node T_181095 = eq(T_181094, T_181063)
    infer accessor T_181096 = idxs[UInt<5>(16)]
    node T_181097 = eq(T_181096, T_181063)
    infer accessor T_181098 = idxs[UInt<5>(17)]
    node T_181099 = eq(T_181098, T_181063)
    infer accessor T_181100 = idxs[UInt<5>(18)]
    node T_181101 = eq(T_181100, T_181063)
    infer accessor T_181102 = idxs[UInt<5>(19)]
    node T_181103 = eq(T_181102, T_181063)
    infer accessor T_181104 = idxs[UInt<5>(20)]
    node T_181105 = eq(T_181104, T_181063)
    infer accessor T_181106 = idxs[UInt<5>(21)]
    node T_181107 = eq(T_181106, T_181063)
    infer accessor T_181108 = idxs[UInt<5>(22)]
    node T_181109 = eq(T_181108, T_181063)
    infer accessor T_181110 = idxs[UInt<5>(23)]
    node T_181111 = eq(T_181110, T_181063)
    infer accessor T_181112 = idxs[UInt<5>(24)]
    node T_181113 = eq(T_181112, T_181063)
    infer accessor T_181114 = idxs[UInt<5>(25)]
    node T_181115 = eq(T_181114, T_181063)
    infer accessor T_181116 = idxs[UInt<5>(26)]
    node T_181117 = eq(T_181116, T_181063)
    infer accessor T_181118 = idxs[UInt<5>(27)]
    node T_181119 = eq(T_181118, T_181063)
    infer accessor T_181120 = idxs[UInt<5>(28)]
    node T_181121 = eq(T_181120, T_181063)
    infer accessor T_181122 = idxs[UInt<5>(29)]
    node T_181123 = eq(T_181122, T_181063)
    infer accessor T_181124 = idxs[UInt<5>(30)]
    node T_181125 = eq(T_181124, T_181063)
    infer accessor T_181126 = idxs[UInt<5>(31)]
    node T_181127 = eq(T_181126, T_181063)
    infer accessor T_181128 = idxs[UInt<6>(32)]
    node T_181129 = eq(T_181128, T_181063)
    infer accessor T_181130 = idxs[UInt<6>(33)]
    node T_181131 = eq(T_181130, T_181063)
    infer accessor T_181132 = idxs[UInt<6>(34)]
    node T_181133 = eq(T_181132, T_181063)
    infer accessor T_181134 = idxs[UInt<6>(35)]
    node T_181135 = eq(T_181134, T_181063)
    infer accessor T_181136 = idxs[UInt<6>(36)]
    node T_181137 = eq(T_181136, T_181063)
    infer accessor T_181138 = idxs[UInt<6>(37)]
    node T_181139 = eq(T_181138, T_181063)
    infer accessor T_181140 = idxs[UInt<6>(38)]
    node T_181141 = eq(T_181140, T_181063)
    infer accessor T_181142 = idxs[UInt<6>(39)]
    node T_181143 = eq(T_181142, T_181063)
    infer accessor T_181144 = idxs[UInt<6>(40)]
    node T_181145 = eq(T_181144, T_181063)
    infer accessor T_181146 = idxs[UInt<6>(41)]
    node T_181147 = eq(T_181146, T_181063)
    infer accessor T_181148 = idxs[UInt<6>(42)]
    node T_181149 = eq(T_181148, T_181063)
    infer accessor T_181150 = idxs[UInt<6>(43)]
    node T_181151 = eq(T_181150, T_181063)
    infer accessor T_181152 = idxs[UInt<6>(44)]
    node T_181153 = eq(T_181152, T_181063)
    infer accessor T_181154 = idxs[UInt<6>(45)]
    node T_181155 = eq(T_181154, T_181063)
    infer accessor T_181156 = idxs[UInt<6>(46)]
    node T_181157 = eq(T_181156, T_181063)
    infer accessor T_181158 = idxs[UInt<6>(47)]
    node T_181159 = eq(T_181158, T_181063)
    infer accessor T_181160 = idxs[UInt<6>(48)]
    node T_181161 = eq(T_181160, T_181063)
    infer accessor T_181162 = idxs[UInt<6>(49)]
    node T_181163 = eq(T_181162, T_181063)
    infer accessor T_181164 = idxs[UInt<6>(50)]
    node T_181165 = eq(T_181164, T_181063)
    infer accessor T_181166 = idxs[UInt<6>(51)]
    node T_181167 = eq(T_181166, T_181063)
    infer accessor T_181168 = idxs[UInt<6>(52)]
    node T_181169 = eq(T_181168, T_181063)
    infer accessor T_181170 = idxs[UInt<6>(53)]
    node T_181171 = eq(T_181170, T_181063)
    infer accessor T_181172 = idxs[UInt<6>(54)]
    node T_181173 = eq(T_181172, T_181063)
    infer accessor T_181174 = idxs[UInt<6>(55)]
    node T_181175 = eq(T_181174, T_181063)
    infer accessor T_181176 = idxs[UInt<6>(56)]
    node T_181177 = eq(T_181176, T_181063)
    infer accessor T_181178 = idxs[UInt<6>(57)]
    node T_181179 = eq(T_181178, T_181063)
    infer accessor T_181180 = idxs[UInt<6>(58)]
    node T_181181 = eq(T_181180, T_181063)
    infer accessor T_181182 = idxs[UInt<6>(59)]
    node T_181183 = eq(T_181182, T_181063)
    infer accessor T_181184 = idxs[UInt<6>(60)]
    node T_181185 = eq(T_181184, T_181063)
    infer accessor T_181186 = idxs[UInt<6>(61)]
    node T_181187 = eq(T_181186, T_181063)
    wire T_181188 : UInt<1>[62]
    T_181188[0] := T_181065
    T_181188[1] := T_181067
    T_181188[2] := T_181069
    T_181188[3] := T_181071
    T_181188[4] := T_181073
    T_181188[5] := T_181075
    T_181188[6] := T_181077
    T_181188[7] := T_181079
    T_181188[8] := T_181081
    T_181188[9] := T_181083
    T_181188[10] := T_181085
    T_181188[11] := T_181087
    T_181188[12] := T_181089
    T_181188[13] := T_181091
    T_181188[14] := T_181093
    T_181188[15] := T_181095
    T_181188[16] := T_181097
    T_181188[17] := T_181099
    T_181188[18] := T_181101
    T_181188[19] := T_181103
    T_181188[20] := T_181105
    T_181188[21] := T_181107
    T_181188[22] := T_181109
    T_181188[23] := T_181111
    T_181188[24] := T_181113
    T_181188[25] := T_181115
    T_181188[26] := T_181117
    T_181188[27] := T_181119
    T_181188[28] := T_181121
    T_181188[29] := T_181123
    T_181188[30] := T_181125
    T_181188[31] := T_181127
    T_181188[32] := T_181129
    T_181188[33] := T_181131
    T_181188[34] := T_181133
    T_181188[35] := T_181135
    T_181188[36] := T_181137
    T_181188[37] := T_181139
    T_181188[38] := T_181141
    T_181188[39] := T_181143
    T_181188[40] := T_181145
    T_181188[41] := T_181147
    T_181188[42] := T_181149
    T_181188[43] := T_181151
    T_181188[44] := T_181153
    T_181188[45] := T_181155
    T_181188[46] := T_181157
    T_181188[47] := T_181159
    T_181188[48] := T_181161
    T_181188[49] := T_181163
    T_181188[50] := T_181165
    T_181188[51] := T_181167
    T_181188[52] := T_181169
    T_181188[53] := T_181171
    T_181188[54] := T_181173
    T_181188[55] := T_181175
    T_181188[56] := T_181177
    T_181188[57] := T_181179
    T_181188[58] := T_181181
    T_181188[59] := T_181183
    T_181188[60] := T_181185
    T_181188[61] := T_181187
    node T_181189 = cat(T_181188[60], T_181188[59])
    node T_181190 = cat(T_181188[61], T_181189)
    node T_181191 = cat(T_181188[58], T_181188[57])
    node T_181192 = cat(T_181188[56], T_181188[55])
    node T_181193 = cat(T_181191, T_181192)
    node T_181194 = cat(T_181190, T_181193)
    node T_181195 = cat(T_181188[54], T_181188[53])
    node T_181196 = cat(T_181188[52], T_181188[51])
    node T_181197 = cat(T_181195, T_181196)
    node T_181198 = cat(T_181188[50], T_181188[49])
    node T_181199 = cat(T_181188[48], T_181188[47])
    node T_181200 = cat(T_181198, T_181199)
    node T_181201 = cat(T_181197, T_181200)
    node T_181202 = cat(T_181194, T_181201)
    node T_181203 = cat(T_181188[46], T_181188[45])
    node T_181204 = cat(T_181188[44], T_181188[43])
    node T_181205 = cat(T_181203, T_181204)
    node T_181206 = cat(T_181188[42], T_181188[41])
    node T_181207 = cat(T_181188[40], T_181188[39])
    node T_181208 = cat(T_181206, T_181207)
    node T_181209 = cat(T_181205, T_181208)
    node T_181210 = cat(T_181188[38], T_181188[37])
    node T_181211 = cat(T_181188[36], T_181188[35])
    node T_181212 = cat(T_181210, T_181211)
    node T_181213 = cat(T_181188[34], T_181188[33])
    node T_181214 = cat(T_181188[32], T_181188[31])
    node T_181215 = cat(T_181213, T_181214)
    node T_181216 = cat(T_181212, T_181215)
    node T_181217 = cat(T_181209, T_181216)
    node T_181218 = cat(T_181202, T_181217)
    node T_181219 = cat(T_181188[29], T_181188[28])
    node T_181220 = cat(T_181188[30], T_181219)
    node T_181221 = cat(T_181188[27], T_181188[26])
    node T_181222 = cat(T_181188[25], T_181188[24])
    node T_181223 = cat(T_181221, T_181222)
    node T_181224 = cat(T_181220, T_181223)
    node T_181225 = cat(T_181188[23], T_181188[22])
    node T_181226 = cat(T_181188[21], T_181188[20])
    node T_181227 = cat(T_181225, T_181226)
    node T_181228 = cat(T_181188[19], T_181188[18])
    node T_181229 = cat(T_181188[17], T_181188[16])
    node T_181230 = cat(T_181228, T_181229)
    node T_181231 = cat(T_181227, T_181230)
    node T_181232 = cat(T_181224, T_181231)
    node T_181233 = cat(T_181188[15], T_181188[14])
    node T_181234 = cat(T_181188[13], T_181188[12])
    node T_181235 = cat(T_181233, T_181234)
    node T_181236 = cat(T_181188[11], T_181188[10])
    node T_181237 = cat(T_181188[9], T_181188[8])
    node T_181238 = cat(T_181236, T_181237)
    node T_181239 = cat(T_181235, T_181238)
    node T_181240 = cat(T_181188[7], T_181188[6])
    node T_181241 = cat(T_181188[5], T_181188[4])
    node T_181242 = cat(T_181240, T_181241)
    node T_181243 = cat(T_181188[3], T_181188[2])
    node T_181244 = cat(T_181188[1], T_181188[0])
    node T_181245 = cat(T_181243, T_181244)
    node T_181246 = cat(T_181242, T_181245)
    node T_181247 = cat(T_181239, T_181246)
    node T_181248 = cat(T_181232, T_181247)
    node T_181249 = cat(T_181218, T_181248)
    node T_181250 = and(T_180672, pageHit)
    node T_181251 = and(T_180675, pageHit)
    node T_181252 = and(T_180678, pageHit)
    node T_181253 = and(T_180681, pageHit)
    node T_181254 = and(T_180684, pageHit)
    node T_181255 = and(T_180687, pageHit)
    node T_181256 = and(T_180690, pageHit)
    node T_181257 = and(T_180693, pageHit)
    node T_181258 = and(T_180696, pageHit)
    node T_181259 = and(T_180699, pageHit)
    node T_181260 = and(T_180702, pageHit)
    node T_181261 = and(T_180705, pageHit)
    node T_181262 = and(T_180708, pageHit)
    node T_181263 = and(T_180711, pageHit)
    node T_181264 = and(T_180714, pageHit)
    node T_181265 = and(T_180717, pageHit)
    node T_181266 = and(T_180720, pageHit)
    node T_181267 = and(T_180723, pageHit)
    node T_181268 = and(T_180726, pageHit)
    node T_181269 = and(T_180729, pageHit)
    node T_181270 = and(T_180732, pageHit)
    node T_181271 = and(T_180735, pageHit)
    node T_181272 = and(T_180738, pageHit)
    node T_181273 = and(T_180741, pageHit)
    node T_181274 = and(T_180744, pageHit)
    node T_181275 = and(T_180747, pageHit)
    node T_181276 = and(T_180750, pageHit)
    node T_181277 = and(T_180753, pageHit)
    node T_181278 = and(T_180756, pageHit)
    node T_181279 = and(T_180759, pageHit)
    node T_181280 = and(T_180762, pageHit)
    node T_181281 = and(T_180765, pageHit)
    node T_181282 = and(T_180768, pageHit)
    node T_181283 = and(T_180771, pageHit)
    node T_181284 = and(T_180774, pageHit)
    node T_181285 = and(T_180777, pageHit)
    node T_181286 = and(T_180780, pageHit)
    node T_181287 = and(T_180783, pageHit)
    node T_181288 = and(T_180786, pageHit)
    node T_181289 = and(T_180789, pageHit)
    node T_181290 = and(T_180792, pageHit)
    node T_181291 = and(T_180795, pageHit)
    node T_181292 = and(T_180798, pageHit)
    node T_181293 = and(T_180801, pageHit)
    node T_181294 = and(T_180804, pageHit)
    node T_181295 = and(T_180807, pageHit)
    node T_181296 = and(T_180810, pageHit)
    node T_181297 = and(T_180813, pageHit)
    node T_181298 = and(T_180816, pageHit)
    node T_181299 = and(T_180819, pageHit)
    node T_181300 = and(T_180822, pageHit)
    node T_181301 = and(T_180825, pageHit)
    node T_181302 = and(T_180828, pageHit)
    node T_181303 = and(T_180831, pageHit)
    node T_181304 = and(T_180834, pageHit)
    node T_181305 = and(T_180837, pageHit)
    node T_181306 = and(T_180840, pageHit)
    node T_181307 = and(T_180843, pageHit)
    node T_181308 = and(T_180846, pageHit)
    node T_181309 = and(T_180849, pageHit)
    node T_181310 = and(T_180852, pageHit)
    node T_181311 = and(T_180855, pageHit)
    node T_181312 = eq(T_181250, UInt<1>(0))
    node T_181313 = eq(T_181312, UInt<1>(0))
    node T_181314 = eq(T_181251, UInt<1>(0))
    node T_181315 = eq(T_181314, UInt<1>(0))
    node T_181316 = eq(T_181252, UInt<1>(0))
    node T_181317 = eq(T_181316, UInt<1>(0))
    node T_181318 = eq(T_181253, UInt<1>(0))
    node T_181319 = eq(T_181318, UInt<1>(0))
    node T_181320 = eq(T_181254, UInt<1>(0))
    node T_181321 = eq(T_181320, UInt<1>(0))
    node T_181322 = eq(T_181255, UInt<1>(0))
    node T_181323 = eq(T_181322, UInt<1>(0))
    node T_181324 = eq(T_181256, UInt<1>(0))
    node T_181325 = eq(T_181324, UInt<1>(0))
    node T_181326 = eq(T_181257, UInt<1>(0))
    node T_181327 = eq(T_181326, UInt<1>(0))
    node T_181328 = eq(T_181258, UInt<1>(0))
    node T_181329 = eq(T_181328, UInt<1>(0))
    node T_181330 = eq(T_181259, UInt<1>(0))
    node T_181331 = eq(T_181330, UInt<1>(0))
    node T_181332 = eq(T_181260, UInt<1>(0))
    node T_181333 = eq(T_181332, UInt<1>(0))
    node T_181334 = eq(T_181261, UInt<1>(0))
    node T_181335 = eq(T_181334, UInt<1>(0))
    node T_181336 = eq(T_181262, UInt<1>(0))
    node T_181337 = eq(T_181336, UInt<1>(0))
    node T_181338 = eq(T_181263, UInt<1>(0))
    node T_181339 = eq(T_181338, UInt<1>(0))
    node T_181340 = eq(T_181264, UInt<1>(0))
    node T_181341 = eq(T_181340, UInt<1>(0))
    node T_181342 = eq(T_181265, UInt<1>(0))
    node T_181343 = eq(T_181342, UInt<1>(0))
    node T_181344 = eq(T_181266, UInt<1>(0))
    node T_181345 = eq(T_181344, UInt<1>(0))
    node T_181346 = eq(T_181267, UInt<1>(0))
    node T_181347 = eq(T_181346, UInt<1>(0))
    node T_181348 = eq(T_181268, UInt<1>(0))
    node T_181349 = eq(T_181348, UInt<1>(0))
    node T_181350 = eq(T_181269, UInt<1>(0))
    node T_181351 = eq(T_181350, UInt<1>(0))
    node T_181352 = eq(T_181270, UInt<1>(0))
    node T_181353 = eq(T_181352, UInt<1>(0))
    node T_181354 = eq(T_181271, UInt<1>(0))
    node T_181355 = eq(T_181354, UInt<1>(0))
    node T_181356 = eq(T_181272, UInt<1>(0))
    node T_181357 = eq(T_181356, UInt<1>(0))
    node T_181358 = eq(T_181273, UInt<1>(0))
    node T_181359 = eq(T_181358, UInt<1>(0))
    node T_181360 = eq(T_181274, UInt<1>(0))
    node T_181361 = eq(T_181360, UInt<1>(0))
    node T_181362 = eq(T_181275, UInt<1>(0))
    node T_181363 = eq(T_181362, UInt<1>(0))
    node T_181364 = eq(T_181276, UInt<1>(0))
    node T_181365 = eq(T_181364, UInt<1>(0))
    node T_181366 = eq(T_181277, UInt<1>(0))
    node T_181367 = eq(T_181366, UInt<1>(0))
    node T_181368 = eq(T_181278, UInt<1>(0))
    node T_181369 = eq(T_181368, UInt<1>(0))
    node T_181370 = eq(T_181279, UInt<1>(0))
    node T_181371 = eq(T_181370, UInt<1>(0))
    node T_181372 = eq(T_181280, UInt<1>(0))
    node T_181373 = eq(T_181372, UInt<1>(0))
    node T_181374 = eq(T_181281, UInt<1>(0))
    node T_181375 = eq(T_181374, UInt<1>(0))
    node T_181376 = eq(T_181282, UInt<1>(0))
    node T_181377 = eq(T_181376, UInt<1>(0))
    node T_181378 = eq(T_181283, UInt<1>(0))
    node T_181379 = eq(T_181378, UInt<1>(0))
    node T_181380 = eq(T_181284, UInt<1>(0))
    node T_181381 = eq(T_181380, UInt<1>(0))
    node T_181382 = eq(T_181285, UInt<1>(0))
    node T_181383 = eq(T_181382, UInt<1>(0))
    node T_181384 = eq(T_181286, UInt<1>(0))
    node T_181385 = eq(T_181384, UInt<1>(0))
    node T_181386 = eq(T_181287, UInt<1>(0))
    node T_181387 = eq(T_181386, UInt<1>(0))
    node T_181388 = eq(T_181288, UInt<1>(0))
    node T_181389 = eq(T_181388, UInt<1>(0))
    node T_181390 = eq(T_181289, UInt<1>(0))
    node T_181391 = eq(T_181390, UInt<1>(0))
    node T_181392 = eq(T_181290, UInt<1>(0))
    node T_181393 = eq(T_181392, UInt<1>(0))
    node T_181394 = eq(T_181291, UInt<1>(0))
    node T_181395 = eq(T_181394, UInt<1>(0))
    node T_181396 = eq(T_181292, UInt<1>(0))
    node T_181397 = eq(T_181396, UInt<1>(0))
    node T_181398 = eq(T_181293, UInt<1>(0))
    node T_181399 = eq(T_181398, UInt<1>(0))
    node T_181400 = eq(T_181294, UInt<1>(0))
    node T_181401 = eq(T_181400, UInt<1>(0))
    node T_181402 = eq(T_181295, UInt<1>(0))
    node T_181403 = eq(T_181402, UInt<1>(0))
    node T_181404 = eq(T_181296, UInt<1>(0))
    node T_181405 = eq(T_181404, UInt<1>(0))
    node T_181406 = eq(T_181297, UInt<1>(0))
    node T_181407 = eq(T_181406, UInt<1>(0))
    node T_181408 = eq(T_181298, UInt<1>(0))
    node T_181409 = eq(T_181408, UInt<1>(0))
    node T_181410 = eq(T_181299, UInt<1>(0))
    node T_181411 = eq(T_181410, UInt<1>(0))
    node T_181412 = eq(T_181300, UInt<1>(0))
    node T_181413 = eq(T_181412, UInt<1>(0))
    node T_181414 = eq(T_181301, UInt<1>(0))
    node T_181415 = eq(T_181414, UInt<1>(0))
    node T_181416 = eq(T_181302, UInt<1>(0))
    node T_181417 = eq(T_181416, UInt<1>(0))
    node T_181418 = eq(T_181303, UInt<1>(0))
    node T_181419 = eq(T_181418, UInt<1>(0))
    node T_181420 = eq(T_181304, UInt<1>(0))
    node T_181421 = eq(T_181420, UInt<1>(0))
    node T_181422 = eq(T_181305, UInt<1>(0))
    node T_181423 = eq(T_181422, UInt<1>(0))
    node T_181424 = eq(T_181306, UInt<1>(0))
    node T_181425 = eq(T_181424, UInt<1>(0))
    node T_181426 = eq(T_181307, UInt<1>(0))
    node T_181427 = eq(T_181426, UInt<1>(0))
    node T_181428 = eq(T_181308, UInt<1>(0))
    node T_181429 = eq(T_181428, UInt<1>(0))
    node T_181430 = eq(T_181309, UInt<1>(0))
    node T_181431 = eq(T_181430, UInt<1>(0))
    node T_181432 = eq(T_181310, UInt<1>(0))
    node T_181433 = eq(T_181432, UInt<1>(0))
    node T_181434 = eq(T_181311, UInt<1>(0))
    node T_181435 = eq(T_181434, UInt<1>(0))
    wire T_181436 : UInt<1>[62]
    T_181436[0] := T_181313
    T_181436[1] := T_181315
    T_181436[2] := T_181317
    T_181436[3] := T_181319
    T_181436[4] := T_181321
    T_181436[5] := T_181323
    T_181436[6] := T_181325
    T_181436[7] := T_181327
    T_181436[8] := T_181329
    T_181436[9] := T_181331
    T_181436[10] := T_181333
    T_181436[11] := T_181335
    T_181436[12] := T_181337
    T_181436[13] := T_181339
    T_181436[14] := T_181341
    T_181436[15] := T_181343
    T_181436[16] := T_181345
    T_181436[17] := T_181347
    T_181436[18] := T_181349
    T_181436[19] := T_181351
    T_181436[20] := T_181353
    T_181436[21] := T_181355
    T_181436[22] := T_181357
    T_181436[23] := T_181359
    T_181436[24] := T_181361
    T_181436[25] := T_181363
    T_181436[26] := T_181365
    T_181436[27] := T_181367
    T_181436[28] := T_181369
    T_181436[29] := T_181371
    T_181436[30] := T_181373
    T_181436[31] := T_181375
    T_181436[32] := T_181377
    T_181436[33] := T_181379
    T_181436[34] := T_181381
    T_181436[35] := T_181383
    T_181436[36] := T_181385
    T_181436[37] := T_181387
    T_181436[38] := T_181389
    T_181436[39] := T_181391
    T_181436[40] := T_181393
    T_181436[41] := T_181395
    T_181436[42] := T_181397
    T_181436[43] := T_181399
    T_181436[44] := T_181401
    T_181436[45] := T_181403
    T_181436[46] := T_181405
    T_181436[47] := T_181407
    T_181436[48] := T_181409
    T_181436[49] := T_181411
    T_181436[50] := T_181413
    T_181436[51] := T_181415
    T_181436[52] := T_181417
    T_181436[53] := T_181419
    T_181436[54] := T_181421
    T_181436[55] := T_181423
    T_181436[56] := T_181425
    T_181436[57] := T_181427
    T_181436[58] := T_181429
    T_181436[59] := T_181431
    T_181436[60] := T_181433
    T_181436[61] := T_181435
    node T_181437 = cat(T_181436[60], T_181436[59])
    node T_181438 = cat(T_181436[61], T_181437)
    node T_181439 = cat(T_181436[58], T_181436[57])
    node T_181440 = cat(T_181436[56], T_181436[55])
    node T_181441 = cat(T_181439, T_181440)
    node T_181442 = cat(T_181438, T_181441)
    node T_181443 = cat(T_181436[54], T_181436[53])
    node T_181444 = cat(T_181436[52], T_181436[51])
    node T_181445 = cat(T_181443, T_181444)
    node T_181446 = cat(T_181436[50], T_181436[49])
    node T_181447 = cat(T_181436[48], T_181436[47])
    node T_181448 = cat(T_181446, T_181447)
    node T_181449 = cat(T_181445, T_181448)
    node T_181450 = cat(T_181442, T_181449)
    node T_181451 = cat(T_181436[46], T_181436[45])
    node T_181452 = cat(T_181436[44], T_181436[43])
    node T_181453 = cat(T_181451, T_181452)
    node T_181454 = cat(T_181436[42], T_181436[41])
    node T_181455 = cat(T_181436[40], T_181436[39])
    node T_181456 = cat(T_181454, T_181455)
    node T_181457 = cat(T_181453, T_181456)
    node T_181458 = cat(T_181436[38], T_181436[37])
    node T_181459 = cat(T_181436[36], T_181436[35])
    node T_181460 = cat(T_181458, T_181459)
    node T_181461 = cat(T_181436[34], T_181436[33])
    node T_181462 = cat(T_181436[32], T_181436[31])
    node T_181463 = cat(T_181461, T_181462)
    node T_181464 = cat(T_181460, T_181463)
    node T_181465 = cat(T_181457, T_181464)
    node T_181466 = cat(T_181450, T_181465)
    node T_181467 = cat(T_181436[29], T_181436[28])
    node T_181468 = cat(T_181436[30], T_181467)
    node T_181469 = cat(T_181436[27], T_181436[26])
    node T_181470 = cat(T_181436[25], T_181436[24])
    node T_181471 = cat(T_181469, T_181470)
    node T_181472 = cat(T_181468, T_181471)
    node T_181473 = cat(T_181436[23], T_181436[22])
    node T_181474 = cat(T_181436[21], T_181436[20])
    node T_181475 = cat(T_181473, T_181474)
    node T_181476 = cat(T_181436[19], T_181436[18])
    node T_181477 = cat(T_181436[17], T_181436[16])
    node T_181478 = cat(T_181476, T_181477)
    node T_181479 = cat(T_181475, T_181478)
    node T_181480 = cat(T_181472, T_181479)
    node T_181481 = cat(T_181436[15], T_181436[14])
    node T_181482 = cat(T_181436[13], T_181436[12])
    node T_181483 = cat(T_181481, T_181482)
    node T_181484 = cat(T_181436[11], T_181436[10])
    node T_181485 = cat(T_181436[9], T_181436[8])
    node T_181486 = cat(T_181484, T_181485)
    node T_181487 = cat(T_181483, T_181486)
    node T_181488 = cat(T_181436[7], T_181436[6])
    node T_181489 = cat(T_181436[5], T_181436[4])
    node T_181490 = cat(T_181488, T_181489)
    node T_181491 = cat(T_181436[3], T_181436[2])
    node T_181492 = cat(T_181436[1], T_181436[0])
    node T_181493 = cat(T_181491, T_181492)
    node T_181494 = cat(T_181490, T_181493)
    node T_181495 = cat(T_181487, T_181494)
    node T_181496 = cat(T_181480, T_181495)
    node T_181497 = cat(T_181466, T_181496)
    node T_181498 = bit(T_181249, 0)
    node T_181499 = and(idxValid[0], T_181498)
    node T_181500 = bit(T_181497, 0)
    node T_181501 = and(T_181499, T_181500)
    node T_181502 = bit(T_181249, 1)
    node T_181503 = and(idxValid[1], T_181502)
    node T_181504 = bit(T_181497, 1)
    node T_181505 = and(T_181503, T_181504)
    node T_181506 = bit(T_181249, 2)
    node T_181507 = and(idxValid[2], T_181506)
    node T_181508 = bit(T_181497, 2)
    node T_181509 = and(T_181507, T_181508)
    node T_181510 = bit(T_181249, 3)
    node T_181511 = and(idxValid[3], T_181510)
    node T_181512 = bit(T_181497, 3)
    node T_181513 = and(T_181511, T_181512)
    node T_181514 = bit(T_181249, 4)
    node T_181515 = and(idxValid[4], T_181514)
    node T_181516 = bit(T_181497, 4)
    node T_181517 = and(T_181515, T_181516)
    node T_181518 = bit(T_181249, 5)
    node T_181519 = and(idxValid[5], T_181518)
    node T_181520 = bit(T_181497, 5)
    node T_181521 = and(T_181519, T_181520)
    node T_181522 = bit(T_181249, 6)
    node T_181523 = and(idxValid[6], T_181522)
    node T_181524 = bit(T_181497, 6)
    node T_181525 = and(T_181523, T_181524)
    node T_181526 = bit(T_181249, 7)
    node T_181527 = and(idxValid[7], T_181526)
    node T_181528 = bit(T_181497, 7)
    node T_181529 = and(T_181527, T_181528)
    node T_181530 = bit(T_181249, 8)
    node T_181531 = and(idxValid[8], T_181530)
    node T_181532 = bit(T_181497, 8)
    node T_181533 = and(T_181531, T_181532)
    node T_181534 = bit(T_181249, 9)
    node T_181535 = and(idxValid[9], T_181534)
    node T_181536 = bit(T_181497, 9)
    node T_181537 = and(T_181535, T_181536)
    node T_181538 = bit(T_181249, 10)
    node T_181539 = and(idxValid[10], T_181538)
    node T_181540 = bit(T_181497, 10)
    node T_181541 = and(T_181539, T_181540)
    node T_181542 = bit(T_181249, 11)
    node T_181543 = and(idxValid[11], T_181542)
    node T_181544 = bit(T_181497, 11)
    node T_181545 = and(T_181543, T_181544)
    node T_181546 = bit(T_181249, 12)
    node T_181547 = and(idxValid[12], T_181546)
    node T_181548 = bit(T_181497, 12)
    node T_181549 = and(T_181547, T_181548)
    node T_181550 = bit(T_181249, 13)
    node T_181551 = and(idxValid[13], T_181550)
    node T_181552 = bit(T_181497, 13)
    node T_181553 = and(T_181551, T_181552)
    node T_181554 = bit(T_181249, 14)
    node T_181555 = and(idxValid[14], T_181554)
    node T_181556 = bit(T_181497, 14)
    node T_181557 = and(T_181555, T_181556)
    node T_181558 = bit(T_181249, 15)
    node T_181559 = and(idxValid[15], T_181558)
    node T_181560 = bit(T_181497, 15)
    node T_181561 = and(T_181559, T_181560)
    node T_181562 = bit(T_181249, 16)
    node T_181563 = and(idxValid[16], T_181562)
    node T_181564 = bit(T_181497, 16)
    node T_181565 = and(T_181563, T_181564)
    node T_181566 = bit(T_181249, 17)
    node T_181567 = and(idxValid[17], T_181566)
    node T_181568 = bit(T_181497, 17)
    node T_181569 = and(T_181567, T_181568)
    node T_181570 = bit(T_181249, 18)
    node T_181571 = and(idxValid[18], T_181570)
    node T_181572 = bit(T_181497, 18)
    node T_181573 = and(T_181571, T_181572)
    node T_181574 = bit(T_181249, 19)
    node T_181575 = and(idxValid[19], T_181574)
    node T_181576 = bit(T_181497, 19)
    node T_181577 = and(T_181575, T_181576)
    node T_181578 = bit(T_181249, 20)
    node T_181579 = and(idxValid[20], T_181578)
    node T_181580 = bit(T_181497, 20)
    node T_181581 = and(T_181579, T_181580)
    node T_181582 = bit(T_181249, 21)
    node T_181583 = and(idxValid[21], T_181582)
    node T_181584 = bit(T_181497, 21)
    node T_181585 = and(T_181583, T_181584)
    node T_181586 = bit(T_181249, 22)
    node T_181587 = and(idxValid[22], T_181586)
    node T_181588 = bit(T_181497, 22)
    node T_181589 = and(T_181587, T_181588)
    node T_181590 = bit(T_181249, 23)
    node T_181591 = and(idxValid[23], T_181590)
    node T_181592 = bit(T_181497, 23)
    node T_181593 = and(T_181591, T_181592)
    node T_181594 = bit(T_181249, 24)
    node T_181595 = and(idxValid[24], T_181594)
    node T_181596 = bit(T_181497, 24)
    node T_181597 = and(T_181595, T_181596)
    node T_181598 = bit(T_181249, 25)
    node T_181599 = and(idxValid[25], T_181598)
    node T_181600 = bit(T_181497, 25)
    node T_181601 = and(T_181599, T_181600)
    node T_181602 = bit(T_181249, 26)
    node T_181603 = and(idxValid[26], T_181602)
    node T_181604 = bit(T_181497, 26)
    node T_181605 = and(T_181603, T_181604)
    node T_181606 = bit(T_181249, 27)
    node T_181607 = and(idxValid[27], T_181606)
    node T_181608 = bit(T_181497, 27)
    node T_181609 = and(T_181607, T_181608)
    node T_181610 = bit(T_181249, 28)
    node T_181611 = and(idxValid[28], T_181610)
    node T_181612 = bit(T_181497, 28)
    node T_181613 = and(T_181611, T_181612)
    node T_181614 = bit(T_181249, 29)
    node T_181615 = and(idxValid[29], T_181614)
    node T_181616 = bit(T_181497, 29)
    node T_181617 = and(T_181615, T_181616)
    node T_181618 = bit(T_181249, 30)
    node T_181619 = and(idxValid[30], T_181618)
    node T_181620 = bit(T_181497, 30)
    node T_181621 = and(T_181619, T_181620)
    node T_181622 = bit(T_181249, 31)
    node T_181623 = and(idxValid[31], T_181622)
    node T_181624 = bit(T_181497, 31)
    node T_181625 = and(T_181623, T_181624)
    node T_181626 = bit(T_181249, 32)
    node T_181627 = and(idxValid[32], T_181626)
    node T_181628 = bit(T_181497, 32)
    node T_181629 = and(T_181627, T_181628)
    node T_181630 = bit(T_181249, 33)
    node T_181631 = and(idxValid[33], T_181630)
    node T_181632 = bit(T_181497, 33)
    node T_181633 = and(T_181631, T_181632)
    node T_181634 = bit(T_181249, 34)
    node T_181635 = and(idxValid[34], T_181634)
    node T_181636 = bit(T_181497, 34)
    node T_181637 = and(T_181635, T_181636)
    node T_181638 = bit(T_181249, 35)
    node T_181639 = and(idxValid[35], T_181638)
    node T_181640 = bit(T_181497, 35)
    node T_181641 = and(T_181639, T_181640)
    node T_181642 = bit(T_181249, 36)
    node T_181643 = and(idxValid[36], T_181642)
    node T_181644 = bit(T_181497, 36)
    node T_181645 = and(T_181643, T_181644)
    node T_181646 = bit(T_181249, 37)
    node T_181647 = and(idxValid[37], T_181646)
    node T_181648 = bit(T_181497, 37)
    node T_181649 = and(T_181647, T_181648)
    node T_181650 = bit(T_181249, 38)
    node T_181651 = and(idxValid[38], T_181650)
    node T_181652 = bit(T_181497, 38)
    node T_181653 = and(T_181651, T_181652)
    node T_181654 = bit(T_181249, 39)
    node T_181655 = and(idxValid[39], T_181654)
    node T_181656 = bit(T_181497, 39)
    node T_181657 = and(T_181655, T_181656)
    node T_181658 = bit(T_181249, 40)
    node T_181659 = and(idxValid[40], T_181658)
    node T_181660 = bit(T_181497, 40)
    node T_181661 = and(T_181659, T_181660)
    node T_181662 = bit(T_181249, 41)
    node T_181663 = and(idxValid[41], T_181662)
    node T_181664 = bit(T_181497, 41)
    node T_181665 = and(T_181663, T_181664)
    node T_181666 = bit(T_181249, 42)
    node T_181667 = and(idxValid[42], T_181666)
    node T_181668 = bit(T_181497, 42)
    node T_181669 = and(T_181667, T_181668)
    node T_181670 = bit(T_181249, 43)
    node T_181671 = and(idxValid[43], T_181670)
    node T_181672 = bit(T_181497, 43)
    node T_181673 = and(T_181671, T_181672)
    node T_181674 = bit(T_181249, 44)
    node T_181675 = and(idxValid[44], T_181674)
    node T_181676 = bit(T_181497, 44)
    node T_181677 = and(T_181675, T_181676)
    node T_181678 = bit(T_181249, 45)
    node T_181679 = and(idxValid[45], T_181678)
    node T_181680 = bit(T_181497, 45)
    node T_181681 = and(T_181679, T_181680)
    node T_181682 = bit(T_181249, 46)
    node T_181683 = and(idxValid[46], T_181682)
    node T_181684 = bit(T_181497, 46)
    node T_181685 = and(T_181683, T_181684)
    node T_181686 = bit(T_181249, 47)
    node T_181687 = and(idxValid[47], T_181686)
    node T_181688 = bit(T_181497, 47)
    node T_181689 = and(T_181687, T_181688)
    node T_181690 = bit(T_181249, 48)
    node T_181691 = and(idxValid[48], T_181690)
    node T_181692 = bit(T_181497, 48)
    node T_181693 = and(T_181691, T_181692)
    node T_181694 = bit(T_181249, 49)
    node T_181695 = and(idxValid[49], T_181694)
    node T_181696 = bit(T_181497, 49)
    node T_181697 = and(T_181695, T_181696)
    node T_181698 = bit(T_181249, 50)
    node T_181699 = and(idxValid[50], T_181698)
    node T_181700 = bit(T_181497, 50)
    node T_181701 = and(T_181699, T_181700)
    node T_181702 = bit(T_181249, 51)
    node T_181703 = and(idxValid[51], T_181702)
    node T_181704 = bit(T_181497, 51)
    node T_181705 = and(T_181703, T_181704)
    node T_181706 = bit(T_181249, 52)
    node T_181707 = and(idxValid[52], T_181706)
    node T_181708 = bit(T_181497, 52)
    node T_181709 = and(T_181707, T_181708)
    node T_181710 = bit(T_181249, 53)
    node T_181711 = and(idxValid[53], T_181710)
    node T_181712 = bit(T_181497, 53)
    node T_181713 = and(T_181711, T_181712)
    node T_181714 = bit(T_181249, 54)
    node T_181715 = and(idxValid[54], T_181714)
    node T_181716 = bit(T_181497, 54)
    node T_181717 = and(T_181715, T_181716)
    node T_181718 = bit(T_181249, 55)
    node T_181719 = and(idxValid[55], T_181718)
    node T_181720 = bit(T_181497, 55)
    node T_181721 = and(T_181719, T_181720)
    node T_181722 = bit(T_181249, 56)
    node T_181723 = and(idxValid[56], T_181722)
    node T_181724 = bit(T_181497, 56)
    node T_181725 = and(T_181723, T_181724)
    node T_181726 = bit(T_181249, 57)
    node T_181727 = and(idxValid[57], T_181726)
    node T_181728 = bit(T_181497, 57)
    node T_181729 = and(T_181727, T_181728)
    node T_181730 = bit(T_181249, 58)
    node T_181731 = and(idxValid[58], T_181730)
    node T_181732 = bit(T_181497, 58)
    node T_181733 = and(T_181731, T_181732)
    node T_181734 = bit(T_181249, 59)
    node T_181735 = and(idxValid[59], T_181734)
    node T_181736 = bit(T_181497, 59)
    node T_181737 = and(T_181735, T_181736)
    node T_181738 = bit(T_181249, 60)
    node T_181739 = and(idxValid[60], T_181738)
    node T_181740 = bit(T_181497, 60)
    node T_181741 = and(T_181739, T_181740)
    node T_181742 = bit(T_181249, 61)
    node T_181743 = and(idxValid[61], T_181742)
    node T_181744 = bit(T_181497, 61)
    node T_181745 = and(T_181743, T_181744)
    wire hits : UInt<1>[62]
    hits[0] := T_181501
    hits[1] := T_181505
    hits[2] := T_181509
    hits[3] := T_181513
    hits[4] := T_181517
    hits[5] := T_181521
    hits[6] := T_181525
    hits[7] := T_181529
    hits[8] := T_181533
    hits[9] := T_181537
    hits[10] := T_181541
    hits[11] := T_181545
    hits[12] := T_181549
    hits[13] := T_181553
    hits[14] := T_181557
    hits[15] := T_181561
    hits[16] := T_181565
    hits[17] := T_181569
    hits[18] := T_181573
    hits[19] := T_181577
    hits[20] := T_181581
    hits[21] := T_181585
    hits[22] := T_181589
    hits[23] := T_181593
    hits[24] := T_181597
    hits[25] := T_181601
    hits[26] := T_181605
    hits[27] := T_181609
    hits[28] := T_181613
    hits[29] := T_181617
    hits[30] := T_181621
    hits[31] := T_181625
    hits[32] := T_181629
    hits[33] := T_181633
    hits[34] := T_181637
    hits[35] := T_181641
    hits[36] := T_181645
    hits[37] := T_181649
    hits[38] := T_181653
    hits[39] := T_181657
    hits[40] := T_181661
    hits[41] := T_181665
    hits[42] := T_181669
    hits[43] := T_181673
    hits[44] := T_181677
    hits[45] := T_181681
    hits[46] := T_181685
    hits[47] := T_181689
    hits[48] := T_181693
    hits[49] := T_181697
    hits[50] := T_181701
    hits[51] := T_181705
    hits[52] := T_181709
    hits[53] := T_181713
    hits[54] := T_181717
    hits[55] := T_181721
    hits[56] := T_181725
    hits[57] := T_181729
    hits[58] := T_181733
    hits[59] := T_181737
    hits[60] := T_181741
    hits[61] := T_181745
    node T_181746 = shr(r_btb_update.bits.pc, 12)
    infer accessor T_181747 = pages[UInt<1>(0)]
    node T_181748 = eq(T_181747, T_181746)
    infer accessor T_181749 = pages[UInt<1>(1)]
    node T_181750 = eq(T_181749, T_181746)
    infer accessor T_181751 = pages[UInt<2>(2)]
    node T_181752 = eq(T_181751, T_181746)
    infer accessor T_181753 = pages[UInt<2>(3)]
    node T_181754 = eq(T_181753, T_181746)
    infer accessor T_181755 = pages[UInt<3>(4)]
    node T_181756 = eq(T_181755, T_181746)
    infer accessor T_181757 = pages[UInt<3>(5)]
    node T_181758 = eq(T_181757, T_181746)
    wire T_181759 : UInt<1>[6]
    T_181759[0] := T_181748
    T_181759[1] := T_181750
    T_181759[2] := T_181752
    T_181759[3] := T_181754
    T_181759[4] := T_181756
    T_181759[5] := T_181758
    node T_181760 = cat(T_181759[4], T_181759[3])
    node T_181761 = cat(T_181759[5], T_181760)
    node T_181762 = cat(T_181759[1], T_181759[0])
    node T_181763 = cat(T_181759[2], T_181762)
    node T_181764 = cat(T_181761, T_181763)
    node updatePageHit = and(T_181764, pageValid)
    node T_181765 = bits(r_btb_update.bits.pc, 11, 0)
    infer accessor T_181766 = idxs[UInt<1>(0)]
    node T_181767 = eq(T_181766, T_181765)
    infer accessor T_181768 = idxs[UInt<1>(1)]
    node T_181769 = eq(T_181768, T_181765)
    infer accessor T_181770 = idxs[UInt<2>(2)]
    node T_181771 = eq(T_181770, T_181765)
    infer accessor T_181772 = idxs[UInt<2>(3)]
    node T_181773 = eq(T_181772, T_181765)
    infer accessor T_181774 = idxs[UInt<3>(4)]
    node T_181775 = eq(T_181774, T_181765)
    infer accessor T_181776 = idxs[UInt<3>(5)]
    node T_181777 = eq(T_181776, T_181765)
    infer accessor T_181778 = idxs[UInt<3>(6)]
    node T_181779 = eq(T_181778, T_181765)
    infer accessor T_181780 = idxs[UInt<3>(7)]
    node T_181781 = eq(T_181780, T_181765)
    infer accessor T_181782 = idxs[UInt<4>(8)]
    node T_181783 = eq(T_181782, T_181765)
    infer accessor T_181784 = idxs[UInt<4>(9)]
    node T_181785 = eq(T_181784, T_181765)
    infer accessor T_181786 = idxs[UInt<4>(10)]
    node T_181787 = eq(T_181786, T_181765)
    infer accessor T_181788 = idxs[UInt<4>(11)]
    node T_181789 = eq(T_181788, T_181765)
    infer accessor T_181790 = idxs[UInt<4>(12)]
    node T_181791 = eq(T_181790, T_181765)
    infer accessor T_181792 = idxs[UInt<4>(13)]
    node T_181793 = eq(T_181792, T_181765)
    infer accessor T_181794 = idxs[UInt<4>(14)]
    node T_181795 = eq(T_181794, T_181765)
    infer accessor T_181796 = idxs[UInt<4>(15)]
    node T_181797 = eq(T_181796, T_181765)
    infer accessor T_181798 = idxs[UInt<5>(16)]
    node T_181799 = eq(T_181798, T_181765)
    infer accessor T_181800 = idxs[UInt<5>(17)]
    node T_181801 = eq(T_181800, T_181765)
    infer accessor T_181802 = idxs[UInt<5>(18)]
    node T_181803 = eq(T_181802, T_181765)
    infer accessor T_181804 = idxs[UInt<5>(19)]
    node T_181805 = eq(T_181804, T_181765)
    infer accessor T_181806 = idxs[UInt<5>(20)]
    node T_181807 = eq(T_181806, T_181765)
    infer accessor T_181808 = idxs[UInt<5>(21)]
    node T_181809 = eq(T_181808, T_181765)
    infer accessor T_181810 = idxs[UInt<5>(22)]
    node T_181811 = eq(T_181810, T_181765)
    infer accessor T_181812 = idxs[UInt<5>(23)]
    node T_181813 = eq(T_181812, T_181765)
    infer accessor T_181814 = idxs[UInt<5>(24)]
    node T_181815 = eq(T_181814, T_181765)
    infer accessor T_181816 = idxs[UInt<5>(25)]
    node T_181817 = eq(T_181816, T_181765)
    infer accessor T_181818 = idxs[UInt<5>(26)]
    node T_181819 = eq(T_181818, T_181765)
    infer accessor T_181820 = idxs[UInt<5>(27)]
    node T_181821 = eq(T_181820, T_181765)
    infer accessor T_181822 = idxs[UInt<5>(28)]
    node T_181823 = eq(T_181822, T_181765)
    infer accessor T_181824 = idxs[UInt<5>(29)]
    node T_181825 = eq(T_181824, T_181765)
    infer accessor T_181826 = idxs[UInt<5>(30)]
    node T_181827 = eq(T_181826, T_181765)
    infer accessor T_181828 = idxs[UInt<5>(31)]
    node T_181829 = eq(T_181828, T_181765)
    infer accessor T_181830 = idxs[UInt<6>(32)]
    node T_181831 = eq(T_181830, T_181765)
    infer accessor T_181832 = idxs[UInt<6>(33)]
    node T_181833 = eq(T_181832, T_181765)
    infer accessor T_181834 = idxs[UInt<6>(34)]
    node T_181835 = eq(T_181834, T_181765)
    infer accessor T_181836 = idxs[UInt<6>(35)]
    node T_181837 = eq(T_181836, T_181765)
    infer accessor T_181838 = idxs[UInt<6>(36)]
    node T_181839 = eq(T_181838, T_181765)
    infer accessor T_181840 = idxs[UInt<6>(37)]
    node T_181841 = eq(T_181840, T_181765)
    infer accessor T_181842 = idxs[UInt<6>(38)]
    node T_181843 = eq(T_181842, T_181765)
    infer accessor T_181844 = idxs[UInt<6>(39)]
    node T_181845 = eq(T_181844, T_181765)
    infer accessor T_181846 = idxs[UInt<6>(40)]
    node T_181847 = eq(T_181846, T_181765)
    infer accessor T_181848 = idxs[UInt<6>(41)]
    node T_181849 = eq(T_181848, T_181765)
    infer accessor T_181850 = idxs[UInt<6>(42)]
    node T_181851 = eq(T_181850, T_181765)
    infer accessor T_181852 = idxs[UInt<6>(43)]
    node T_181853 = eq(T_181852, T_181765)
    infer accessor T_181854 = idxs[UInt<6>(44)]
    node T_181855 = eq(T_181854, T_181765)
    infer accessor T_181856 = idxs[UInt<6>(45)]
    node T_181857 = eq(T_181856, T_181765)
    infer accessor T_181858 = idxs[UInt<6>(46)]
    node T_181859 = eq(T_181858, T_181765)
    infer accessor T_181860 = idxs[UInt<6>(47)]
    node T_181861 = eq(T_181860, T_181765)
    infer accessor T_181862 = idxs[UInt<6>(48)]
    node T_181863 = eq(T_181862, T_181765)
    infer accessor T_181864 = idxs[UInt<6>(49)]
    node T_181865 = eq(T_181864, T_181765)
    infer accessor T_181866 = idxs[UInt<6>(50)]
    node T_181867 = eq(T_181866, T_181765)
    infer accessor T_181868 = idxs[UInt<6>(51)]
    node T_181869 = eq(T_181868, T_181765)
    infer accessor T_181870 = idxs[UInt<6>(52)]
    node T_181871 = eq(T_181870, T_181765)
    infer accessor T_181872 = idxs[UInt<6>(53)]
    node T_181873 = eq(T_181872, T_181765)
    infer accessor T_181874 = idxs[UInt<6>(54)]
    node T_181875 = eq(T_181874, T_181765)
    infer accessor T_181876 = idxs[UInt<6>(55)]
    node T_181877 = eq(T_181876, T_181765)
    infer accessor T_181878 = idxs[UInt<6>(56)]
    node T_181879 = eq(T_181878, T_181765)
    infer accessor T_181880 = idxs[UInt<6>(57)]
    node T_181881 = eq(T_181880, T_181765)
    infer accessor T_181882 = idxs[UInt<6>(58)]
    node T_181883 = eq(T_181882, T_181765)
    infer accessor T_181884 = idxs[UInt<6>(59)]
    node T_181885 = eq(T_181884, T_181765)
    infer accessor T_181886 = idxs[UInt<6>(60)]
    node T_181887 = eq(T_181886, T_181765)
    infer accessor T_181888 = idxs[UInt<6>(61)]
    node T_181889 = eq(T_181888, T_181765)
    wire T_181890 : UInt<1>[62]
    T_181890[0] := T_181767
    T_181890[1] := T_181769
    T_181890[2] := T_181771
    T_181890[3] := T_181773
    T_181890[4] := T_181775
    T_181890[5] := T_181777
    T_181890[6] := T_181779
    T_181890[7] := T_181781
    T_181890[8] := T_181783
    T_181890[9] := T_181785
    T_181890[10] := T_181787
    T_181890[11] := T_181789
    T_181890[12] := T_181791
    T_181890[13] := T_181793
    T_181890[14] := T_181795
    T_181890[15] := T_181797
    T_181890[16] := T_181799
    T_181890[17] := T_181801
    T_181890[18] := T_181803
    T_181890[19] := T_181805
    T_181890[20] := T_181807
    T_181890[21] := T_181809
    T_181890[22] := T_181811
    T_181890[23] := T_181813
    T_181890[24] := T_181815
    T_181890[25] := T_181817
    T_181890[26] := T_181819
    T_181890[27] := T_181821
    T_181890[28] := T_181823
    T_181890[29] := T_181825
    T_181890[30] := T_181827
    T_181890[31] := T_181829
    T_181890[32] := T_181831
    T_181890[33] := T_181833
    T_181890[34] := T_181835
    T_181890[35] := T_181837
    T_181890[36] := T_181839
    T_181890[37] := T_181841
    T_181890[38] := T_181843
    T_181890[39] := T_181845
    T_181890[40] := T_181847
    T_181890[41] := T_181849
    T_181890[42] := T_181851
    T_181890[43] := T_181853
    T_181890[44] := T_181855
    T_181890[45] := T_181857
    T_181890[46] := T_181859
    T_181890[47] := T_181861
    T_181890[48] := T_181863
    T_181890[49] := T_181865
    T_181890[50] := T_181867
    T_181890[51] := T_181869
    T_181890[52] := T_181871
    T_181890[53] := T_181873
    T_181890[54] := T_181875
    T_181890[55] := T_181877
    T_181890[56] := T_181879
    T_181890[57] := T_181881
    T_181890[58] := T_181883
    T_181890[59] := T_181885
    T_181890[60] := T_181887
    T_181890[61] := T_181889
    node T_181891 = cat(T_181890[60], T_181890[59])
    node T_181892 = cat(T_181890[61], T_181891)
    node T_181893 = cat(T_181890[58], T_181890[57])
    node T_181894 = cat(T_181890[56], T_181890[55])
    node T_181895 = cat(T_181893, T_181894)
    node T_181896 = cat(T_181892, T_181895)
    node T_181897 = cat(T_181890[54], T_181890[53])
    node T_181898 = cat(T_181890[52], T_181890[51])
    node T_181899 = cat(T_181897, T_181898)
    node T_181900 = cat(T_181890[50], T_181890[49])
    node T_181901 = cat(T_181890[48], T_181890[47])
    node T_181902 = cat(T_181900, T_181901)
    node T_181903 = cat(T_181899, T_181902)
    node T_181904 = cat(T_181896, T_181903)
    node T_181905 = cat(T_181890[46], T_181890[45])
    node T_181906 = cat(T_181890[44], T_181890[43])
    node T_181907 = cat(T_181905, T_181906)
    node T_181908 = cat(T_181890[42], T_181890[41])
    node T_181909 = cat(T_181890[40], T_181890[39])
    node T_181910 = cat(T_181908, T_181909)
    node T_181911 = cat(T_181907, T_181910)
    node T_181912 = cat(T_181890[38], T_181890[37])
    node T_181913 = cat(T_181890[36], T_181890[35])
    node T_181914 = cat(T_181912, T_181913)
    node T_181915 = cat(T_181890[34], T_181890[33])
    node T_181916 = cat(T_181890[32], T_181890[31])
    node T_181917 = cat(T_181915, T_181916)
    node T_181918 = cat(T_181914, T_181917)
    node T_181919 = cat(T_181911, T_181918)
    node T_181920 = cat(T_181904, T_181919)
    node T_181921 = cat(T_181890[29], T_181890[28])
    node T_181922 = cat(T_181890[30], T_181921)
    node T_181923 = cat(T_181890[27], T_181890[26])
    node T_181924 = cat(T_181890[25], T_181890[24])
    node T_181925 = cat(T_181923, T_181924)
    node T_181926 = cat(T_181922, T_181925)
    node T_181927 = cat(T_181890[23], T_181890[22])
    node T_181928 = cat(T_181890[21], T_181890[20])
    node T_181929 = cat(T_181927, T_181928)
    node T_181930 = cat(T_181890[19], T_181890[18])
    node T_181931 = cat(T_181890[17], T_181890[16])
    node T_181932 = cat(T_181930, T_181931)
    node T_181933 = cat(T_181929, T_181932)
    node T_181934 = cat(T_181926, T_181933)
    node T_181935 = cat(T_181890[15], T_181890[14])
    node T_181936 = cat(T_181890[13], T_181890[12])
    node T_181937 = cat(T_181935, T_181936)
    node T_181938 = cat(T_181890[11], T_181890[10])
    node T_181939 = cat(T_181890[9], T_181890[8])
    node T_181940 = cat(T_181938, T_181939)
    node T_181941 = cat(T_181937, T_181940)
    node T_181942 = cat(T_181890[7], T_181890[6])
    node T_181943 = cat(T_181890[5], T_181890[4])
    node T_181944 = cat(T_181942, T_181943)
    node T_181945 = cat(T_181890[3], T_181890[2])
    node T_181946 = cat(T_181890[1], T_181890[0])
    node T_181947 = cat(T_181945, T_181946)
    node T_181948 = cat(T_181944, T_181947)
    node T_181949 = cat(T_181941, T_181948)
    node T_181950 = cat(T_181934, T_181949)
    node T_181951 = cat(T_181920, T_181950)
    node T_181952 = and(T_180672, updatePageHit)
    node T_181953 = and(T_180675, updatePageHit)
    node T_181954 = and(T_180678, updatePageHit)
    node T_181955 = and(T_180681, updatePageHit)
    node T_181956 = and(T_180684, updatePageHit)
    node T_181957 = and(T_180687, updatePageHit)
    node T_181958 = and(T_180690, updatePageHit)
    node T_181959 = and(T_180693, updatePageHit)
    node T_181960 = and(T_180696, updatePageHit)
    node T_181961 = and(T_180699, updatePageHit)
    node T_181962 = and(T_180702, updatePageHit)
    node T_181963 = and(T_180705, updatePageHit)
    node T_181964 = and(T_180708, updatePageHit)
    node T_181965 = and(T_180711, updatePageHit)
    node T_181966 = and(T_180714, updatePageHit)
    node T_181967 = and(T_180717, updatePageHit)
    node T_181968 = and(T_180720, updatePageHit)
    node T_181969 = and(T_180723, updatePageHit)
    node T_181970 = and(T_180726, updatePageHit)
    node T_181971 = and(T_180729, updatePageHit)
    node T_181972 = and(T_180732, updatePageHit)
    node T_181973 = and(T_180735, updatePageHit)
    node T_181974 = and(T_180738, updatePageHit)
    node T_181975 = and(T_180741, updatePageHit)
    node T_181976 = and(T_180744, updatePageHit)
    node T_181977 = and(T_180747, updatePageHit)
    node T_181978 = and(T_180750, updatePageHit)
    node T_181979 = and(T_180753, updatePageHit)
    node T_181980 = and(T_180756, updatePageHit)
    node T_181981 = and(T_180759, updatePageHit)
    node T_181982 = and(T_180762, updatePageHit)
    node T_181983 = and(T_180765, updatePageHit)
    node T_181984 = and(T_180768, updatePageHit)
    node T_181985 = and(T_180771, updatePageHit)
    node T_181986 = and(T_180774, updatePageHit)
    node T_181987 = and(T_180777, updatePageHit)
    node T_181988 = and(T_180780, updatePageHit)
    node T_181989 = and(T_180783, updatePageHit)
    node T_181990 = and(T_180786, updatePageHit)
    node T_181991 = and(T_180789, updatePageHit)
    node T_181992 = and(T_180792, updatePageHit)
    node T_181993 = and(T_180795, updatePageHit)
    node T_181994 = and(T_180798, updatePageHit)
    node T_181995 = and(T_180801, updatePageHit)
    node T_181996 = and(T_180804, updatePageHit)
    node T_181997 = and(T_180807, updatePageHit)
    node T_181998 = and(T_180810, updatePageHit)
    node T_181999 = and(T_180813, updatePageHit)
    node T_182000 = and(T_180816, updatePageHit)
    node T_182001 = and(T_180819, updatePageHit)
    node T_182002 = and(T_180822, updatePageHit)
    node T_182003 = and(T_180825, updatePageHit)
    node T_182004 = and(T_180828, updatePageHit)
    node T_182005 = and(T_180831, updatePageHit)
    node T_182006 = and(T_180834, updatePageHit)
    node T_182007 = and(T_180837, updatePageHit)
    node T_182008 = and(T_180840, updatePageHit)
    node T_182009 = and(T_180843, updatePageHit)
    node T_182010 = and(T_180846, updatePageHit)
    node T_182011 = and(T_180849, updatePageHit)
    node T_182012 = and(T_180852, updatePageHit)
    node T_182013 = and(T_180855, updatePageHit)
    node T_182014 = eq(T_181952, UInt<1>(0))
    node T_182015 = eq(T_182014, UInt<1>(0))
    node T_182016 = eq(T_181953, UInt<1>(0))
    node T_182017 = eq(T_182016, UInt<1>(0))
    node T_182018 = eq(T_181954, UInt<1>(0))
    node T_182019 = eq(T_182018, UInt<1>(0))
    node T_182020 = eq(T_181955, UInt<1>(0))
    node T_182021 = eq(T_182020, UInt<1>(0))
    node T_182022 = eq(T_181956, UInt<1>(0))
    node T_182023 = eq(T_182022, UInt<1>(0))
    node T_182024 = eq(T_181957, UInt<1>(0))
    node T_182025 = eq(T_182024, UInt<1>(0))
    node T_182026 = eq(T_181958, UInt<1>(0))
    node T_182027 = eq(T_182026, UInt<1>(0))
    node T_182028 = eq(T_181959, UInt<1>(0))
    node T_182029 = eq(T_182028, UInt<1>(0))
    node T_182030 = eq(T_181960, UInt<1>(0))
    node T_182031 = eq(T_182030, UInt<1>(0))
    node T_182032 = eq(T_181961, UInt<1>(0))
    node T_182033 = eq(T_182032, UInt<1>(0))
    node T_182034 = eq(T_181962, UInt<1>(0))
    node T_182035 = eq(T_182034, UInt<1>(0))
    node T_182036 = eq(T_181963, UInt<1>(0))
    node T_182037 = eq(T_182036, UInt<1>(0))
    node T_182038 = eq(T_181964, UInt<1>(0))
    node T_182039 = eq(T_182038, UInt<1>(0))
    node T_182040 = eq(T_181965, UInt<1>(0))
    node T_182041 = eq(T_182040, UInt<1>(0))
    node T_182042 = eq(T_181966, UInt<1>(0))
    node T_182043 = eq(T_182042, UInt<1>(0))
    node T_182044 = eq(T_181967, UInt<1>(0))
    node T_182045 = eq(T_182044, UInt<1>(0))
    node T_182046 = eq(T_181968, UInt<1>(0))
    node T_182047 = eq(T_182046, UInt<1>(0))
    node T_182048 = eq(T_181969, UInt<1>(0))
    node T_182049 = eq(T_182048, UInt<1>(0))
    node T_182050 = eq(T_181970, UInt<1>(0))
    node T_182051 = eq(T_182050, UInt<1>(0))
    node T_182052 = eq(T_181971, UInt<1>(0))
    node T_182053 = eq(T_182052, UInt<1>(0))
    node T_182054 = eq(T_181972, UInt<1>(0))
    node T_182055 = eq(T_182054, UInt<1>(0))
    node T_182056 = eq(T_181973, UInt<1>(0))
    node T_182057 = eq(T_182056, UInt<1>(0))
    node T_182058 = eq(T_181974, UInt<1>(0))
    node T_182059 = eq(T_182058, UInt<1>(0))
    node T_182060 = eq(T_181975, UInt<1>(0))
    node T_182061 = eq(T_182060, UInt<1>(0))
    node T_182062 = eq(T_181976, UInt<1>(0))
    node T_182063 = eq(T_182062, UInt<1>(0))
    node T_182064 = eq(T_181977, UInt<1>(0))
    node T_182065 = eq(T_182064, UInt<1>(0))
    node T_182066 = eq(T_181978, UInt<1>(0))
    node T_182067 = eq(T_182066, UInt<1>(0))
    node T_182068 = eq(T_181979, UInt<1>(0))
    node T_182069 = eq(T_182068, UInt<1>(0))
    node T_182070 = eq(T_181980, UInt<1>(0))
    node T_182071 = eq(T_182070, UInt<1>(0))
    node T_182072 = eq(T_181981, UInt<1>(0))
    node T_182073 = eq(T_182072, UInt<1>(0))
    node T_182074 = eq(T_181982, UInt<1>(0))
    node T_182075 = eq(T_182074, UInt<1>(0))
    node T_182076 = eq(T_181983, UInt<1>(0))
    node T_182077 = eq(T_182076, UInt<1>(0))
    node T_182078 = eq(T_181984, UInt<1>(0))
    node T_182079 = eq(T_182078, UInt<1>(0))
    node T_182080 = eq(T_181985, UInt<1>(0))
    node T_182081 = eq(T_182080, UInt<1>(0))
    node T_182082 = eq(T_181986, UInt<1>(0))
    node T_182083 = eq(T_182082, UInt<1>(0))
    node T_182084 = eq(T_181987, UInt<1>(0))
    node T_182085 = eq(T_182084, UInt<1>(0))
    node T_182086 = eq(T_181988, UInt<1>(0))
    node T_182087 = eq(T_182086, UInt<1>(0))
    node T_182088 = eq(T_181989, UInt<1>(0))
    node T_182089 = eq(T_182088, UInt<1>(0))
    node T_182090 = eq(T_181990, UInt<1>(0))
    node T_182091 = eq(T_182090, UInt<1>(0))
    node T_182092 = eq(T_181991, UInt<1>(0))
    node T_182093 = eq(T_182092, UInt<1>(0))
    node T_182094 = eq(T_181992, UInt<1>(0))
    node T_182095 = eq(T_182094, UInt<1>(0))
    node T_182096 = eq(T_181993, UInt<1>(0))
    node T_182097 = eq(T_182096, UInt<1>(0))
    node T_182098 = eq(T_181994, UInt<1>(0))
    node T_182099 = eq(T_182098, UInt<1>(0))
    node T_182100 = eq(T_181995, UInt<1>(0))
    node T_182101 = eq(T_182100, UInt<1>(0))
    node T_182102 = eq(T_181996, UInt<1>(0))
    node T_182103 = eq(T_182102, UInt<1>(0))
    node T_182104 = eq(T_181997, UInt<1>(0))
    node T_182105 = eq(T_182104, UInt<1>(0))
    node T_182106 = eq(T_181998, UInt<1>(0))
    node T_182107 = eq(T_182106, UInt<1>(0))
    node T_182108 = eq(T_181999, UInt<1>(0))
    node T_182109 = eq(T_182108, UInt<1>(0))
    node T_182110 = eq(T_182000, UInt<1>(0))
    node T_182111 = eq(T_182110, UInt<1>(0))
    node T_182112 = eq(T_182001, UInt<1>(0))
    node T_182113 = eq(T_182112, UInt<1>(0))
    node T_182114 = eq(T_182002, UInt<1>(0))
    node T_182115 = eq(T_182114, UInt<1>(0))
    node T_182116 = eq(T_182003, UInt<1>(0))
    node T_182117 = eq(T_182116, UInt<1>(0))
    node T_182118 = eq(T_182004, UInt<1>(0))
    node T_182119 = eq(T_182118, UInt<1>(0))
    node T_182120 = eq(T_182005, UInt<1>(0))
    node T_182121 = eq(T_182120, UInt<1>(0))
    node T_182122 = eq(T_182006, UInt<1>(0))
    node T_182123 = eq(T_182122, UInt<1>(0))
    node T_182124 = eq(T_182007, UInt<1>(0))
    node T_182125 = eq(T_182124, UInt<1>(0))
    node T_182126 = eq(T_182008, UInt<1>(0))
    node T_182127 = eq(T_182126, UInt<1>(0))
    node T_182128 = eq(T_182009, UInt<1>(0))
    node T_182129 = eq(T_182128, UInt<1>(0))
    node T_182130 = eq(T_182010, UInt<1>(0))
    node T_182131 = eq(T_182130, UInt<1>(0))
    node T_182132 = eq(T_182011, UInt<1>(0))
    node T_182133 = eq(T_182132, UInt<1>(0))
    node T_182134 = eq(T_182012, UInt<1>(0))
    node T_182135 = eq(T_182134, UInt<1>(0))
    node T_182136 = eq(T_182013, UInt<1>(0))
    node T_182137 = eq(T_182136, UInt<1>(0))
    wire T_182138 : UInt<1>[62]
    T_182138[0] := T_182015
    T_182138[1] := T_182017
    T_182138[2] := T_182019
    T_182138[3] := T_182021
    T_182138[4] := T_182023
    T_182138[5] := T_182025
    T_182138[6] := T_182027
    T_182138[7] := T_182029
    T_182138[8] := T_182031
    T_182138[9] := T_182033
    T_182138[10] := T_182035
    T_182138[11] := T_182037
    T_182138[12] := T_182039
    T_182138[13] := T_182041
    T_182138[14] := T_182043
    T_182138[15] := T_182045
    T_182138[16] := T_182047
    T_182138[17] := T_182049
    T_182138[18] := T_182051
    T_182138[19] := T_182053
    T_182138[20] := T_182055
    T_182138[21] := T_182057
    T_182138[22] := T_182059
    T_182138[23] := T_182061
    T_182138[24] := T_182063
    T_182138[25] := T_182065
    T_182138[26] := T_182067
    T_182138[27] := T_182069
    T_182138[28] := T_182071
    T_182138[29] := T_182073
    T_182138[30] := T_182075
    T_182138[31] := T_182077
    T_182138[32] := T_182079
    T_182138[33] := T_182081
    T_182138[34] := T_182083
    T_182138[35] := T_182085
    T_182138[36] := T_182087
    T_182138[37] := T_182089
    T_182138[38] := T_182091
    T_182138[39] := T_182093
    T_182138[40] := T_182095
    T_182138[41] := T_182097
    T_182138[42] := T_182099
    T_182138[43] := T_182101
    T_182138[44] := T_182103
    T_182138[45] := T_182105
    T_182138[46] := T_182107
    T_182138[47] := T_182109
    T_182138[48] := T_182111
    T_182138[49] := T_182113
    T_182138[50] := T_182115
    T_182138[51] := T_182117
    T_182138[52] := T_182119
    T_182138[53] := T_182121
    T_182138[54] := T_182123
    T_182138[55] := T_182125
    T_182138[56] := T_182127
    T_182138[57] := T_182129
    T_182138[58] := T_182131
    T_182138[59] := T_182133
    T_182138[60] := T_182135
    T_182138[61] := T_182137
    node T_182139 = cat(T_182138[60], T_182138[59])
    node T_182140 = cat(T_182138[61], T_182139)
    node T_182141 = cat(T_182138[58], T_182138[57])
    node T_182142 = cat(T_182138[56], T_182138[55])
    node T_182143 = cat(T_182141, T_182142)
    node T_182144 = cat(T_182140, T_182143)
    node T_182145 = cat(T_182138[54], T_182138[53])
    node T_182146 = cat(T_182138[52], T_182138[51])
    node T_182147 = cat(T_182145, T_182146)
    node T_182148 = cat(T_182138[50], T_182138[49])
    node T_182149 = cat(T_182138[48], T_182138[47])
    node T_182150 = cat(T_182148, T_182149)
    node T_182151 = cat(T_182147, T_182150)
    node T_182152 = cat(T_182144, T_182151)
    node T_182153 = cat(T_182138[46], T_182138[45])
    node T_182154 = cat(T_182138[44], T_182138[43])
    node T_182155 = cat(T_182153, T_182154)
    node T_182156 = cat(T_182138[42], T_182138[41])
    node T_182157 = cat(T_182138[40], T_182138[39])
    node T_182158 = cat(T_182156, T_182157)
    node T_182159 = cat(T_182155, T_182158)
    node T_182160 = cat(T_182138[38], T_182138[37])
    node T_182161 = cat(T_182138[36], T_182138[35])
    node T_182162 = cat(T_182160, T_182161)
    node T_182163 = cat(T_182138[34], T_182138[33])
    node T_182164 = cat(T_182138[32], T_182138[31])
    node T_182165 = cat(T_182163, T_182164)
    node T_182166 = cat(T_182162, T_182165)
    node T_182167 = cat(T_182159, T_182166)
    node T_182168 = cat(T_182152, T_182167)
    node T_182169 = cat(T_182138[29], T_182138[28])
    node T_182170 = cat(T_182138[30], T_182169)
    node T_182171 = cat(T_182138[27], T_182138[26])
    node T_182172 = cat(T_182138[25], T_182138[24])
    node T_182173 = cat(T_182171, T_182172)
    node T_182174 = cat(T_182170, T_182173)
    node T_182175 = cat(T_182138[23], T_182138[22])
    node T_182176 = cat(T_182138[21], T_182138[20])
    node T_182177 = cat(T_182175, T_182176)
    node T_182178 = cat(T_182138[19], T_182138[18])
    node T_182179 = cat(T_182138[17], T_182138[16])
    node T_182180 = cat(T_182178, T_182179)
    node T_182181 = cat(T_182177, T_182180)
    node T_182182 = cat(T_182174, T_182181)
    node T_182183 = cat(T_182138[15], T_182138[14])
    node T_182184 = cat(T_182138[13], T_182138[12])
    node T_182185 = cat(T_182183, T_182184)
    node T_182186 = cat(T_182138[11], T_182138[10])
    node T_182187 = cat(T_182138[9], T_182138[8])
    node T_182188 = cat(T_182186, T_182187)
    node T_182189 = cat(T_182185, T_182188)
    node T_182190 = cat(T_182138[7], T_182138[6])
    node T_182191 = cat(T_182138[5], T_182138[4])
    node T_182192 = cat(T_182190, T_182191)
    node T_182193 = cat(T_182138[3], T_182138[2])
    node T_182194 = cat(T_182138[1], T_182138[0])
    node T_182195 = cat(T_182193, T_182194)
    node T_182196 = cat(T_182192, T_182195)
    node T_182197 = cat(T_182189, T_182196)
    node T_182198 = cat(T_182182, T_182197)
    node T_182199 = cat(T_182168, T_182198)
    node T_182200 = bit(T_181951, 0)
    node T_182201 = and(idxValid[0], T_182200)
    node T_182202 = bit(T_182199, 0)
    node T_182203 = and(T_182201, T_182202)
    node T_182204 = bit(T_181951, 1)
    node T_182205 = and(idxValid[1], T_182204)
    node T_182206 = bit(T_182199, 1)
    node T_182207 = and(T_182205, T_182206)
    node T_182208 = bit(T_181951, 2)
    node T_182209 = and(idxValid[2], T_182208)
    node T_182210 = bit(T_182199, 2)
    node T_182211 = and(T_182209, T_182210)
    node T_182212 = bit(T_181951, 3)
    node T_182213 = and(idxValid[3], T_182212)
    node T_182214 = bit(T_182199, 3)
    node T_182215 = and(T_182213, T_182214)
    node T_182216 = bit(T_181951, 4)
    node T_182217 = and(idxValid[4], T_182216)
    node T_182218 = bit(T_182199, 4)
    node T_182219 = and(T_182217, T_182218)
    node T_182220 = bit(T_181951, 5)
    node T_182221 = and(idxValid[5], T_182220)
    node T_182222 = bit(T_182199, 5)
    node T_182223 = and(T_182221, T_182222)
    node T_182224 = bit(T_181951, 6)
    node T_182225 = and(idxValid[6], T_182224)
    node T_182226 = bit(T_182199, 6)
    node T_182227 = and(T_182225, T_182226)
    node T_182228 = bit(T_181951, 7)
    node T_182229 = and(idxValid[7], T_182228)
    node T_182230 = bit(T_182199, 7)
    node T_182231 = and(T_182229, T_182230)
    node T_182232 = bit(T_181951, 8)
    node T_182233 = and(idxValid[8], T_182232)
    node T_182234 = bit(T_182199, 8)
    node T_182235 = and(T_182233, T_182234)
    node T_182236 = bit(T_181951, 9)
    node T_182237 = and(idxValid[9], T_182236)
    node T_182238 = bit(T_182199, 9)
    node T_182239 = and(T_182237, T_182238)
    node T_182240 = bit(T_181951, 10)
    node T_182241 = and(idxValid[10], T_182240)
    node T_182242 = bit(T_182199, 10)
    node T_182243 = and(T_182241, T_182242)
    node T_182244 = bit(T_181951, 11)
    node T_182245 = and(idxValid[11], T_182244)
    node T_182246 = bit(T_182199, 11)
    node T_182247 = and(T_182245, T_182246)
    node T_182248 = bit(T_181951, 12)
    node T_182249 = and(idxValid[12], T_182248)
    node T_182250 = bit(T_182199, 12)
    node T_182251 = and(T_182249, T_182250)
    node T_182252 = bit(T_181951, 13)
    node T_182253 = and(idxValid[13], T_182252)
    node T_182254 = bit(T_182199, 13)
    node T_182255 = and(T_182253, T_182254)
    node T_182256 = bit(T_181951, 14)
    node T_182257 = and(idxValid[14], T_182256)
    node T_182258 = bit(T_182199, 14)
    node T_182259 = and(T_182257, T_182258)
    node T_182260 = bit(T_181951, 15)
    node T_182261 = and(idxValid[15], T_182260)
    node T_182262 = bit(T_182199, 15)
    node T_182263 = and(T_182261, T_182262)
    node T_182264 = bit(T_181951, 16)
    node T_182265 = and(idxValid[16], T_182264)
    node T_182266 = bit(T_182199, 16)
    node T_182267 = and(T_182265, T_182266)
    node T_182268 = bit(T_181951, 17)
    node T_182269 = and(idxValid[17], T_182268)
    node T_182270 = bit(T_182199, 17)
    node T_182271 = and(T_182269, T_182270)
    node T_182272 = bit(T_181951, 18)
    node T_182273 = and(idxValid[18], T_182272)
    node T_182274 = bit(T_182199, 18)
    node T_182275 = and(T_182273, T_182274)
    node T_182276 = bit(T_181951, 19)
    node T_182277 = and(idxValid[19], T_182276)
    node T_182278 = bit(T_182199, 19)
    node T_182279 = and(T_182277, T_182278)
    node T_182280 = bit(T_181951, 20)
    node T_182281 = and(idxValid[20], T_182280)
    node T_182282 = bit(T_182199, 20)
    node T_182283 = and(T_182281, T_182282)
    node T_182284 = bit(T_181951, 21)
    node T_182285 = and(idxValid[21], T_182284)
    node T_182286 = bit(T_182199, 21)
    node T_182287 = and(T_182285, T_182286)
    node T_182288 = bit(T_181951, 22)
    node T_182289 = and(idxValid[22], T_182288)
    node T_182290 = bit(T_182199, 22)
    node T_182291 = and(T_182289, T_182290)
    node T_182292 = bit(T_181951, 23)
    node T_182293 = and(idxValid[23], T_182292)
    node T_182294 = bit(T_182199, 23)
    node T_182295 = and(T_182293, T_182294)
    node T_182296 = bit(T_181951, 24)
    node T_182297 = and(idxValid[24], T_182296)
    node T_182298 = bit(T_182199, 24)
    node T_182299 = and(T_182297, T_182298)
    node T_182300 = bit(T_181951, 25)
    node T_182301 = and(idxValid[25], T_182300)
    node T_182302 = bit(T_182199, 25)
    node T_182303 = and(T_182301, T_182302)
    node T_182304 = bit(T_181951, 26)
    node T_182305 = and(idxValid[26], T_182304)
    node T_182306 = bit(T_182199, 26)
    node T_182307 = and(T_182305, T_182306)
    node T_182308 = bit(T_181951, 27)
    node T_182309 = and(idxValid[27], T_182308)
    node T_182310 = bit(T_182199, 27)
    node T_182311 = and(T_182309, T_182310)
    node T_182312 = bit(T_181951, 28)
    node T_182313 = and(idxValid[28], T_182312)
    node T_182314 = bit(T_182199, 28)
    node T_182315 = and(T_182313, T_182314)
    node T_182316 = bit(T_181951, 29)
    node T_182317 = and(idxValid[29], T_182316)
    node T_182318 = bit(T_182199, 29)
    node T_182319 = and(T_182317, T_182318)
    node T_182320 = bit(T_181951, 30)
    node T_182321 = and(idxValid[30], T_182320)
    node T_182322 = bit(T_182199, 30)
    node T_182323 = and(T_182321, T_182322)
    node T_182324 = bit(T_181951, 31)
    node T_182325 = and(idxValid[31], T_182324)
    node T_182326 = bit(T_182199, 31)
    node T_182327 = and(T_182325, T_182326)
    node T_182328 = bit(T_181951, 32)
    node T_182329 = and(idxValid[32], T_182328)
    node T_182330 = bit(T_182199, 32)
    node T_182331 = and(T_182329, T_182330)
    node T_182332 = bit(T_181951, 33)
    node T_182333 = and(idxValid[33], T_182332)
    node T_182334 = bit(T_182199, 33)
    node T_182335 = and(T_182333, T_182334)
    node T_182336 = bit(T_181951, 34)
    node T_182337 = and(idxValid[34], T_182336)
    node T_182338 = bit(T_182199, 34)
    node T_182339 = and(T_182337, T_182338)
    node T_182340 = bit(T_181951, 35)
    node T_182341 = and(idxValid[35], T_182340)
    node T_182342 = bit(T_182199, 35)
    node T_182343 = and(T_182341, T_182342)
    node T_182344 = bit(T_181951, 36)
    node T_182345 = and(idxValid[36], T_182344)
    node T_182346 = bit(T_182199, 36)
    node T_182347 = and(T_182345, T_182346)
    node T_182348 = bit(T_181951, 37)
    node T_182349 = and(idxValid[37], T_182348)
    node T_182350 = bit(T_182199, 37)
    node T_182351 = and(T_182349, T_182350)
    node T_182352 = bit(T_181951, 38)
    node T_182353 = and(idxValid[38], T_182352)
    node T_182354 = bit(T_182199, 38)
    node T_182355 = and(T_182353, T_182354)
    node T_182356 = bit(T_181951, 39)
    node T_182357 = and(idxValid[39], T_182356)
    node T_182358 = bit(T_182199, 39)
    node T_182359 = and(T_182357, T_182358)
    node T_182360 = bit(T_181951, 40)
    node T_182361 = and(idxValid[40], T_182360)
    node T_182362 = bit(T_182199, 40)
    node T_182363 = and(T_182361, T_182362)
    node T_182364 = bit(T_181951, 41)
    node T_182365 = and(idxValid[41], T_182364)
    node T_182366 = bit(T_182199, 41)
    node T_182367 = and(T_182365, T_182366)
    node T_182368 = bit(T_181951, 42)
    node T_182369 = and(idxValid[42], T_182368)
    node T_182370 = bit(T_182199, 42)
    node T_182371 = and(T_182369, T_182370)
    node T_182372 = bit(T_181951, 43)
    node T_182373 = and(idxValid[43], T_182372)
    node T_182374 = bit(T_182199, 43)
    node T_182375 = and(T_182373, T_182374)
    node T_182376 = bit(T_181951, 44)
    node T_182377 = and(idxValid[44], T_182376)
    node T_182378 = bit(T_182199, 44)
    node T_182379 = and(T_182377, T_182378)
    node T_182380 = bit(T_181951, 45)
    node T_182381 = and(idxValid[45], T_182380)
    node T_182382 = bit(T_182199, 45)
    node T_182383 = and(T_182381, T_182382)
    node T_182384 = bit(T_181951, 46)
    node T_182385 = and(idxValid[46], T_182384)
    node T_182386 = bit(T_182199, 46)
    node T_182387 = and(T_182385, T_182386)
    node T_182388 = bit(T_181951, 47)
    node T_182389 = and(idxValid[47], T_182388)
    node T_182390 = bit(T_182199, 47)
    node T_182391 = and(T_182389, T_182390)
    node T_182392 = bit(T_181951, 48)
    node T_182393 = and(idxValid[48], T_182392)
    node T_182394 = bit(T_182199, 48)
    node T_182395 = and(T_182393, T_182394)
    node T_182396 = bit(T_181951, 49)
    node T_182397 = and(idxValid[49], T_182396)
    node T_182398 = bit(T_182199, 49)
    node T_182399 = and(T_182397, T_182398)
    node T_182400 = bit(T_181951, 50)
    node T_182401 = and(idxValid[50], T_182400)
    node T_182402 = bit(T_182199, 50)
    node T_182403 = and(T_182401, T_182402)
    node T_182404 = bit(T_181951, 51)
    node T_182405 = and(idxValid[51], T_182404)
    node T_182406 = bit(T_182199, 51)
    node T_182407 = and(T_182405, T_182406)
    node T_182408 = bit(T_181951, 52)
    node T_182409 = and(idxValid[52], T_182408)
    node T_182410 = bit(T_182199, 52)
    node T_182411 = and(T_182409, T_182410)
    node T_182412 = bit(T_181951, 53)
    node T_182413 = and(idxValid[53], T_182412)
    node T_182414 = bit(T_182199, 53)
    node T_182415 = and(T_182413, T_182414)
    node T_182416 = bit(T_181951, 54)
    node T_182417 = and(idxValid[54], T_182416)
    node T_182418 = bit(T_182199, 54)
    node T_182419 = and(T_182417, T_182418)
    node T_182420 = bit(T_181951, 55)
    node T_182421 = and(idxValid[55], T_182420)
    node T_182422 = bit(T_182199, 55)
    node T_182423 = and(T_182421, T_182422)
    node T_182424 = bit(T_181951, 56)
    node T_182425 = and(idxValid[56], T_182424)
    node T_182426 = bit(T_182199, 56)
    node T_182427 = and(T_182425, T_182426)
    node T_182428 = bit(T_181951, 57)
    node T_182429 = and(idxValid[57], T_182428)
    node T_182430 = bit(T_182199, 57)
    node T_182431 = and(T_182429, T_182430)
    node T_182432 = bit(T_181951, 58)
    node T_182433 = and(idxValid[58], T_182432)
    node T_182434 = bit(T_182199, 58)
    node T_182435 = and(T_182433, T_182434)
    node T_182436 = bit(T_181951, 59)
    node T_182437 = and(idxValid[59], T_182436)
    node T_182438 = bit(T_182199, 59)
    node T_182439 = and(T_182437, T_182438)
    node T_182440 = bit(T_181951, 60)
    node T_182441 = and(idxValid[60], T_182440)
    node T_182442 = bit(T_182199, 60)
    node T_182443 = and(T_182441, T_182442)
    node T_182444 = bit(T_181951, 61)
    node T_182445 = and(idxValid[61], T_182444)
    node T_182446 = bit(T_182199, 61)
    node T_182447 = and(T_182445, T_182446)
    wire updateHits : UInt<1>[62]
    updateHits[0] := T_182203
    updateHits[1] := T_182207
    updateHits[2] := T_182211
    updateHits[3] := T_182215
    updateHits[4] := T_182219
    updateHits[5] := T_182223
    updateHits[6] := T_182227
    updateHits[7] := T_182231
    updateHits[8] := T_182235
    updateHits[9] := T_182239
    updateHits[10] := T_182243
    updateHits[11] := T_182247
    updateHits[12] := T_182251
    updateHits[13] := T_182255
    updateHits[14] := T_182259
    updateHits[15] := T_182263
    updateHits[16] := T_182267
    updateHits[17] := T_182271
    updateHits[18] := T_182275
    updateHits[19] := T_182279
    updateHits[20] := T_182283
    updateHits[21] := T_182287
    updateHits[22] := T_182291
    updateHits[23] := T_182295
    updateHits[24] := T_182299
    updateHits[25] := T_182303
    updateHits[26] := T_182307
    updateHits[27] := T_182311
    updateHits[28] := T_182315
    updateHits[29] := T_182319
    updateHits[30] := T_182323
    updateHits[31] := T_182327
    updateHits[32] := T_182331
    updateHits[33] := T_182335
    updateHits[34] := T_182339
    updateHits[35] := T_182343
    updateHits[36] := T_182347
    updateHits[37] := T_182351
    updateHits[38] := T_182355
    updateHits[39] := T_182359
    updateHits[40] := T_182363
    updateHits[41] := T_182367
    updateHits[42] := T_182371
    updateHits[43] := T_182375
    updateHits[44] := T_182379
    updateHits[45] := T_182383
    updateHits[46] := T_182387
    updateHits[47] := T_182391
    updateHits[48] := T_182395
    updateHits[49] := T_182399
    updateHits[50] := T_182403
    updateHits[51] := T_182407
    updateHits[52] := T_182411
    updateHits[53] := T_182415
    updateHits[54] := T_182419
    updateHits[55] := T_182423
    updateHits[56] := T_182427
    updateHits[57] := T_182431
    updateHits[58] := T_182435
    updateHits[59] := T_182439
    updateHits[60] := T_182443
    updateHits[61] := T_182447
    reg T_182448 : UInt<16>, clk, reset
    onreset T_182448 := UInt<16>(1)
    when r_btb_update.valid : 
      node T_182449 = bit(T_182448, 0)
      node T_182450 = bit(T_182448, 2)
      node T_182451 = xor(T_182449, T_182450)
      node T_182452 = bit(T_182448, 3)
      node T_182453 = xor(T_182451, T_182452)
      node T_182454 = bit(T_182448, 5)
      node T_182455 = xor(T_182453, T_182454)
      node T_182456 = bits(T_182448, 15, 1)
      node T_182457 = cat(T_182455, T_182456)
      T_182448 := T_182457
    node T_182458 = eq(updatePageHit, UInt<1>(0))
    node useUpdatePageHit = eq(T_182458, UInt<1>(0))
    node doIdxPageRepl = eq(useUpdatePageHit, UInt<1>(0))
    wire idxPageRepl : UInt
    wire idxPageUpdateOH : UInt
    idxPageUpdateOH := idxPageRepl
    when useUpdatePageHit : idxPageUpdateOH := updatePageHit
    wire idxPageReplEn : UInt<1>
    idxPageReplEn := UInt<1>(0)
    when doIdxPageRepl : idxPageReplEn := idxPageRepl
    node T_182459 = shr(r_btb_update.bits.pc, 12)
    node T_182460 = shr(req.bits.addr, 12)
    node samePage = eq(T_182459, T_182460)
    node T_182461 = not(idxPageReplEn)
    node T_182462 = and(pageHit, T_182461)
    node T_182463 = eq(T_182462, UInt<1>(0))
    node usePageHit = eq(T_182463, UInt<1>(0))
    node T_182464 = eq(samePage, UInt<1>(0))
    node T_182465 = eq(usePageHit, UInt<1>(0))
    node doTgtPageRepl = and(T_182464, T_182465)
    node T_182466 = bits(idxPageUpdateOH, 4, 0)
    node T_182467 = shl(T_182466, 1)
    node T_182468 = bit(idxPageUpdateOH, 5)
    node T_182469 = or(T_182467, T_182468)
    wire tgtPageRepl : UInt<6>
    tgtPageRepl := T_182469
    when samePage : tgtPageRepl := idxPageUpdateOH
    wire T_182470 : UInt<6>
    T_182470 := tgtPageRepl
    when usePageHit : T_182470 := pageHit
    node T_182471 = bits(T_182470, 5, 4)
    node T_182472 = bits(T_182470, 3, 0)
    node T_182473 = eq(T_182471, UInt<1>(0))
    node T_182474 = eq(T_182473, UInt<1>(0))
    node T_182475 = or(T_182471, T_182472)
    node T_182476 = bits(T_182475, 3, 2)
    node T_182477 = bits(T_182475, 1, 0)
    node T_182478 = eq(T_182476, UInt<1>(0))
    node T_182479 = eq(T_182478, UInt<1>(0))
    node T_182480 = or(T_182476, T_182477)
    node T_182481 = bit(T_182480, 1)
    node T_182482 = cat(T_182479, T_182481)
    node tgtPageUpdate = cat(T_182474, T_182482)
    wire tgtPageReplEn : UInt<1>
    tgtPageReplEn := UInt<1>(0)
    when doTgtPageRepl : tgtPageReplEn := tgtPageRepl
    node doPageRepl = or(doIdxPageRepl, doTgtPageRepl)
    node pageReplEn = or(idxPageReplEn, tgtPageReplEn)
    node T_182483 = and(r_btb_update.valid, doPageRepl)
    reg T_182484 : UInt<3>, clk, reset
    onreset T_182484 := UInt<3>(0)
    when T_182483 : 
      node T_182485 = eq(T_182484, UInt<3>(5))
      node T_182486 = and(UInt<1>(1), T_182485)
      node T_182487 = addw(T_182484, UInt<1>(1))
      wire T_182488 : UInt<3>
      T_182488 := T_182487
      when T_182486 : T_182488 := UInt<1>(0)
      T_182484 := T_182488
    node T_182489 = and(T_182483, T_182485)
    node T_182490 = dshl(UInt<1>(1), T_182484)
    idxPageRepl := T_182490
    when r_btb_update.valid : 
      node T_182491 = eq(req.bits.addr, r_btb_update.bits.target)
      node T_182492 = eq(r_btb_update.bits.prediction.valid, UInt<1>(0))
      reg T_182493 : UInt<6>, clk, reset
      onreset T_182493 := UInt<6>(0)
      when T_182492 : 
        node T_182494 = eq(T_182493, UInt<6>(61))
        node T_182495 = and(UInt<1>(1), T_182494)
        node T_182496 = addw(T_182493, UInt<1>(1))
        wire T_182497 : UInt<6>
        T_182497 := T_182496
        when T_182495 : T_182497 := UInt<1>(0)
        T_182493 := T_182497
      node T_182498 = and(T_182492, T_182494)
      wire T_182499 : UInt<6>
      T_182499 := T_182493
      when r_btb_update.bits.prediction.valid : T_182499 := r_btb_update.bits.prediction.bits.entry
      
      
      node T_182500 = or(T_180672, T_180858)
      node T_182501 = and(pageReplEn, T_182500)
      node T_182502 = eq(T_182501, UInt<1>(0))
      node T_182503 = eq(T_182502, UInt<1>(0))
      when T_182503 : idxValid[0] := UInt<1>(0)
      
      
      node T_182504 = or(T_180675, T_180861)
      node T_182505 = and(pageReplEn, T_182504)
      node T_182506 = eq(T_182505, UInt<1>(0))
      node T_182507 = eq(T_182506, UInt<1>(0))
      when T_182507 : idxValid[1] := UInt<1>(0)
      
      
      node T_182508 = or(T_180678, T_180864)
      node T_182509 = and(pageReplEn, T_182508)
      node T_182510 = eq(T_182509, UInt<1>(0))
      node T_182511 = eq(T_182510, UInt<1>(0))
      when T_182511 : idxValid[2] := UInt<1>(0)
      
      
      node T_182512 = or(T_180681, T_180867)
      node T_182513 = and(pageReplEn, T_182512)
      node T_182514 = eq(T_182513, UInt<1>(0))
      node T_182515 = eq(T_182514, UInt<1>(0))
      when T_182515 : idxValid[3] := UInt<1>(0)
      
      
      node T_182516 = or(T_180684, T_180870)
      node T_182517 = and(pageReplEn, T_182516)
      node T_182518 = eq(T_182517, UInt<1>(0))
      node T_182519 = eq(T_182518, UInt<1>(0))
      when T_182519 : idxValid[4] := UInt<1>(0)
      
      
      node T_182520 = or(T_180687, T_180873)
      node T_182521 = and(pageReplEn, T_182520)
      node T_182522 = eq(T_182521, UInt<1>(0))
      node T_182523 = eq(T_182522, UInt<1>(0))
      when T_182523 : idxValid[5] := UInt<1>(0)
      
      
      node T_182524 = or(T_180690, T_180876)
      node T_182525 = and(pageReplEn, T_182524)
      node T_182526 = eq(T_182525, UInt<1>(0))
      node T_182527 = eq(T_182526, UInt<1>(0))
      when T_182527 : idxValid[6] := UInt<1>(0)
      
      
      node T_182528 = or(T_180693, T_180879)
      node T_182529 = and(pageReplEn, T_182528)
      node T_182530 = eq(T_182529, UInt<1>(0))
      node T_182531 = eq(T_182530, UInt<1>(0))
      when T_182531 : idxValid[7] := UInt<1>(0)
      
      
      node T_182532 = or(T_180696, T_180882)
      node T_182533 = and(pageReplEn, T_182532)
      node T_182534 = eq(T_182533, UInt<1>(0))
      node T_182535 = eq(T_182534, UInt<1>(0))
      when T_182535 : idxValid[8] := UInt<1>(0)
      
      
      node T_182536 = or(T_180699, T_180885)
      node T_182537 = and(pageReplEn, T_182536)
      node T_182538 = eq(T_182537, UInt<1>(0))
      node T_182539 = eq(T_182538, UInt<1>(0))
      when T_182539 : idxValid[9] := UInt<1>(0)
      
      
      node T_182540 = or(T_180702, T_180888)
      node T_182541 = and(pageReplEn, T_182540)
      node T_182542 = eq(T_182541, UInt<1>(0))
      node T_182543 = eq(T_182542, UInt<1>(0))
      when T_182543 : idxValid[10] := UInt<1>(0)
      
      
      node T_182544 = or(T_180705, T_180891)
      node T_182545 = and(pageReplEn, T_182544)
      node T_182546 = eq(T_182545, UInt<1>(0))
      node T_182547 = eq(T_182546, UInt<1>(0))
      when T_182547 : idxValid[11] := UInt<1>(0)
      
      
      node T_182548 = or(T_180708, T_180894)
      node T_182549 = and(pageReplEn, T_182548)
      node T_182550 = eq(T_182549, UInt<1>(0))
      node T_182551 = eq(T_182550, UInt<1>(0))
      when T_182551 : idxValid[12] := UInt<1>(0)
      
      
      node T_182552 = or(T_180711, T_180897)
      node T_182553 = and(pageReplEn, T_182552)
      node T_182554 = eq(T_182553, UInt<1>(0))
      node T_182555 = eq(T_182554, UInt<1>(0))
      when T_182555 : idxValid[13] := UInt<1>(0)
      
      
      node T_182556 = or(T_180714, T_180900)
      node T_182557 = and(pageReplEn, T_182556)
      node T_182558 = eq(T_182557, UInt<1>(0))
      node T_182559 = eq(T_182558, UInt<1>(0))
      when T_182559 : idxValid[14] := UInt<1>(0)
      
      
      node T_182560 = or(T_180717, T_180903)
      node T_182561 = and(pageReplEn, T_182560)
      node T_182562 = eq(T_182561, UInt<1>(0))
      node T_182563 = eq(T_182562, UInt<1>(0))
      when T_182563 : idxValid[15] := UInt<1>(0)
      
      
      node T_182564 = or(T_180720, T_180906)
      node T_182565 = and(pageReplEn, T_182564)
      node T_182566 = eq(T_182565, UInt<1>(0))
      node T_182567 = eq(T_182566, UInt<1>(0))
      when T_182567 : idxValid[16] := UInt<1>(0)
      
      
      node T_182568 = or(T_180723, T_180909)
      node T_182569 = and(pageReplEn, T_182568)
      node T_182570 = eq(T_182569, UInt<1>(0))
      node T_182571 = eq(T_182570, UInt<1>(0))
      when T_182571 : idxValid[17] := UInt<1>(0)
      
      
      node T_182572 = or(T_180726, T_180912)
      node T_182573 = and(pageReplEn, T_182572)
      node T_182574 = eq(T_182573, UInt<1>(0))
      node T_182575 = eq(T_182574, UInt<1>(0))
      when T_182575 : idxValid[18] := UInt<1>(0)
      
      
      node T_182576 = or(T_180729, T_180915)
      node T_182577 = and(pageReplEn, T_182576)
      node T_182578 = eq(T_182577, UInt<1>(0))
      node T_182579 = eq(T_182578, UInt<1>(0))
      when T_182579 : idxValid[19] := UInt<1>(0)
      
      
      node T_182580 = or(T_180732, T_180918)
      node T_182581 = and(pageReplEn, T_182580)
      node T_182582 = eq(T_182581, UInt<1>(0))
      node T_182583 = eq(T_182582, UInt<1>(0))
      when T_182583 : idxValid[20] := UInt<1>(0)
      
      
      node T_182584 = or(T_180735, T_180921)
      node T_182585 = and(pageReplEn, T_182584)
      node T_182586 = eq(T_182585, UInt<1>(0))
      node T_182587 = eq(T_182586, UInt<1>(0))
      when T_182587 : idxValid[21] := UInt<1>(0)
      
      
      node T_182588 = or(T_180738, T_180924)
      node T_182589 = and(pageReplEn, T_182588)
      node T_182590 = eq(T_182589, UInt<1>(0))
      node T_182591 = eq(T_182590, UInt<1>(0))
      when T_182591 : idxValid[22] := UInt<1>(0)
      
      
      node T_182592 = or(T_180741, T_180927)
      node T_182593 = and(pageReplEn, T_182592)
      node T_182594 = eq(T_182593, UInt<1>(0))
      node T_182595 = eq(T_182594, UInt<1>(0))
      when T_182595 : idxValid[23] := UInt<1>(0)
      
      
      node T_182596 = or(T_180744, T_180930)
      node T_182597 = and(pageReplEn, T_182596)
      node T_182598 = eq(T_182597, UInt<1>(0))
      node T_182599 = eq(T_182598, UInt<1>(0))
      when T_182599 : idxValid[24] := UInt<1>(0)
      
      
      node T_182600 = or(T_180747, T_180933)
      node T_182601 = and(pageReplEn, T_182600)
      node T_182602 = eq(T_182601, UInt<1>(0))
      node T_182603 = eq(T_182602, UInt<1>(0))
      when T_182603 : idxValid[25] := UInt<1>(0)
      
      
      node T_182604 = or(T_180750, T_180936)
      node T_182605 = and(pageReplEn, T_182604)
      node T_182606 = eq(T_182605, UInt<1>(0))
      node T_182607 = eq(T_182606, UInt<1>(0))
      when T_182607 : idxValid[26] := UInt<1>(0)
      
      
      node T_182608 = or(T_180753, T_180939)
      node T_182609 = and(pageReplEn, T_182608)
      node T_182610 = eq(T_182609, UInt<1>(0))
      node T_182611 = eq(T_182610, UInt<1>(0))
      when T_182611 : idxValid[27] := UInt<1>(0)
      
      
      node T_182612 = or(T_180756, T_180942)
      node T_182613 = and(pageReplEn, T_182612)
      node T_182614 = eq(T_182613, UInt<1>(0))
      node T_182615 = eq(T_182614, UInt<1>(0))
      when T_182615 : idxValid[28] := UInt<1>(0)
      
      
      node T_182616 = or(T_180759, T_180945)
      node T_182617 = and(pageReplEn, T_182616)
      node T_182618 = eq(T_182617, UInt<1>(0))
      node T_182619 = eq(T_182618, UInt<1>(0))
      when T_182619 : idxValid[29] := UInt<1>(0)
      
      
      node T_182620 = or(T_180762, T_180948)
      node T_182621 = and(pageReplEn, T_182620)
      node T_182622 = eq(T_182621, UInt<1>(0))
      node T_182623 = eq(T_182622, UInt<1>(0))
      when T_182623 : idxValid[30] := UInt<1>(0)
      
      
      node T_182624 = or(T_180765, T_180951)
      node T_182625 = and(pageReplEn, T_182624)
      node T_182626 = eq(T_182625, UInt<1>(0))
      node T_182627 = eq(T_182626, UInt<1>(0))
      when T_182627 : idxValid[31] := UInt<1>(0)
      
      
      node T_182628 = or(T_180768, T_180954)
      node T_182629 = and(pageReplEn, T_182628)
      node T_182630 = eq(T_182629, UInt<1>(0))
      node T_182631 = eq(T_182630, UInt<1>(0))
      when T_182631 : idxValid[32] := UInt<1>(0)
      
      
      node T_182632 = or(T_180771, T_180957)
      node T_182633 = and(pageReplEn, T_182632)
      node T_182634 = eq(T_182633, UInt<1>(0))
      node T_182635 = eq(T_182634, UInt<1>(0))
      when T_182635 : idxValid[33] := UInt<1>(0)
      
      
      node T_182636 = or(T_180774, T_180960)
      node T_182637 = and(pageReplEn, T_182636)
      node T_182638 = eq(T_182637, UInt<1>(0))
      node T_182639 = eq(T_182638, UInt<1>(0))
      when T_182639 : idxValid[34] := UInt<1>(0)
      
      
      node T_182640 = or(T_180777, T_180963)
      node T_182641 = and(pageReplEn, T_182640)
      node T_182642 = eq(T_182641, UInt<1>(0))
      node T_182643 = eq(T_182642, UInt<1>(0))
      when T_182643 : idxValid[35] := UInt<1>(0)
      
      
      node T_182644 = or(T_180780, T_180966)
      node T_182645 = and(pageReplEn, T_182644)
      node T_182646 = eq(T_182645, UInt<1>(0))
      node T_182647 = eq(T_182646, UInt<1>(0))
      when T_182647 : idxValid[36] := UInt<1>(0)
      
      
      node T_182648 = or(T_180783, T_180969)
      node T_182649 = and(pageReplEn, T_182648)
      node T_182650 = eq(T_182649, UInt<1>(0))
      node T_182651 = eq(T_182650, UInt<1>(0))
      when T_182651 : idxValid[37] := UInt<1>(0)
      
      
      node T_182652 = or(T_180786, T_180972)
      node T_182653 = and(pageReplEn, T_182652)
      node T_182654 = eq(T_182653, UInt<1>(0))
      node T_182655 = eq(T_182654, UInt<1>(0))
      when T_182655 : idxValid[38] := UInt<1>(0)
      
      
      node T_182656 = or(T_180789, T_180975)
      node T_182657 = and(pageReplEn, T_182656)
      node T_182658 = eq(T_182657, UInt<1>(0))
      node T_182659 = eq(T_182658, UInt<1>(0))
      when T_182659 : idxValid[39] := UInt<1>(0)
      
      
      node T_182660 = or(T_180792, T_180978)
      node T_182661 = and(pageReplEn, T_182660)
      node T_182662 = eq(T_182661, UInt<1>(0))
      node T_182663 = eq(T_182662, UInt<1>(0))
      when T_182663 : idxValid[40] := UInt<1>(0)
      
      
      node T_182664 = or(T_180795, T_180981)
      node T_182665 = and(pageReplEn, T_182664)
      node T_182666 = eq(T_182665, UInt<1>(0))
      node T_182667 = eq(T_182666, UInt<1>(0))
      when T_182667 : idxValid[41] := UInt<1>(0)
      
      
      node T_182668 = or(T_180798, T_180984)
      node T_182669 = and(pageReplEn, T_182668)
      node T_182670 = eq(T_182669, UInt<1>(0))
      node T_182671 = eq(T_182670, UInt<1>(0))
      when T_182671 : idxValid[42] := UInt<1>(0)
      
      
      node T_182672 = or(T_180801, T_180987)
      node T_182673 = and(pageReplEn, T_182672)
      node T_182674 = eq(T_182673, UInt<1>(0))
      node T_182675 = eq(T_182674, UInt<1>(0))
      when T_182675 : idxValid[43] := UInt<1>(0)
      
      
      node T_182676 = or(T_180804, T_180990)
      node T_182677 = and(pageReplEn, T_182676)
      node T_182678 = eq(T_182677, UInt<1>(0))
      node T_182679 = eq(T_182678, UInt<1>(0))
      when T_182679 : idxValid[44] := UInt<1>(0)
      
      
      node T_182680 = or(T_180807, T_180993)
      node T_182681 = and(pageReplEn, T_182680)
      node T_182682 = eq(T_182681, UInt<1>(0))
      node T_182683 = eq(T_182682, UInt<1>(0))
      when T_182683 : idxValid[45] := UInt<1>(0)
      
      
      node T_182684 = or(T_180810, T_180996)
      node T_182685 = and(pageReplEn, T_182684)
      node T_182686 = eq(T_182685, UInt<1>(0))
      node T_182687 = eq(T_182686, UInt<1>(0))
      when T_182687 : idxValid[46] := UInt<1>(0)
      
      
      node T_182688 = or(T_180813, T_180999)
      node T_182689 = and(pageReplEn, T_182688)
      node T_182690 = eq(T_182689, UInt<1>(0))
      node T_182691 = eq(T_182690, UInt<1>(0))
      when T_182691 : idxValid[47] := UInt<1>(0)
      
      
      node T_182692 = or(T_180816, T_181002)
      node T_182693 = and(pageReplEn, T_182692)
      node T_182694 = eq(T_182693, UInt<1>(0))
      node T_182695 = eq(T_182694, UInt<1>(0))
      when T_182695 : idxValid[48] := UInt<1>(0)
      
      
      node T_182696 = or(T_180819, T_181005)
      node T_182697 = and(pageReplEn, T_182696)
      node T_182698 = eq(T_182697, UInt<1>(0))
      node T_182699 = eq(T_182698, UInt<1>(0))
      when T_182699 : idxValid[49] := UInt<1>(0)
      
      
      node T_182700 = or(T_180822, T_181008)
      node T_182701 = and(pageReplEn, T_182700)
      node T_182702 = eq(T_182701, UInt<1>(0))
      node T_182703 = eq(T_182702, UInt<1>(0))
      when T_182703 : idxValid[50] := UInt<1>(0)
      
      
      node T_182704 = or(T_180825, T_181011)
      node T_182705 = and(pageReplEn, T_182704)
      node T_182706 = eq(T_182705, UInt<1>(0))
      node T_182707 = eq(T_182706, UInt<1>(0))
      when T_182707 : idxValid[51] := UInt<1>(0)
      
      
      node T_182708 = or(T_180828, T_181014)
      node T_182709 = and(pageReplEn, T_182708)
      node T_182710 = eq(T_182709, UInt<1>(0))
      node T_182711 = eq(T_182710, UInt<1>(0))
      when T_182711 : idxValid[52] := UInt<1>(0)
      
      
      node T_182712 = or(T_180831, T_181017)
      node T_182713 = and(pageReplEn, T_182712)
      node T_182714 = eq(T_182713, UInt<1>(0))
      node T_182715 = eq(T_182714, UInt<1>(0))
      when T_182715 : idxValid[53] := UInt<1>(0)
      
      
      node T_182716 = or(T_180834, T_181020)
      node T_182717 = and(pageReplEn, T_182716)
      node T_182718 = eq(T_182717, UInt<1>(0))
      node T_182719 = eq(T_182718, UInt<1>(0))
      when T_182719 : idxValid[54] := UInt<1>(0)
      
      
      node T_182720 = or(T_180837, T_181023)
      node T_182721 = and(pageReplEn, T_182720)
      node T_182722 = eq(T_182721, UInt<1>(0))
      node T_182723 = eq(T_182722, UInt<1>(0))
      when T_182723 : idxValid[55] := UInt<1>(0)
      
      
      node T_182724 = or(T_180840, T_181026)
      node T_182725 = and(pageReplEn, T_182724)
      node T_182726 = eq(T_182725, UInt<1>(0))
      node T_182727 = eq(T_182726, UInt<1>(0))
      when T_182727 : idxValid[56] := UInt<1>(0)
      
      
      node T_182728 = or(T_180843, T_181029)
      node T_182729 = and(pageReplEn, T_182728)
      node T_182730 = eq(T_182729, UInt<1>(0))
      node T_182731 = eq(T_182730, UInt<1>(0))
      when T_182731 : idxValid[57] := UInt<1>(0)
      
      
      node T_182732 = or(T_180846, T_181032)
      node T_182733 = and(pageReplEn, T_182732)
      node T_182734 = eq(T_182733, UInt<1>(0))
      node T_182735 = eq(T_182734, UInt<1>(0))
      when T_182735 : idxValid[58] := UInt<1>(0)
      
      
      node T_182736 = or(T_180849, T_181035)
      node T_182737 = and(pageReplEn, T_182736)
      node T_182738 = eq(T_182737, UInt<1>(0))
      node T_182739 = eq(T_182738, UInt<1>(0))
      when T_182739 : idxValid[59] := UInt<1>(0)
      
      
      node T_182740 = or(T_180852, T_181038)
      node T_182741 = and(pageReplEn, T_182740)
      node T_182742 = eq(T_182741, UInt<1>(0))
      node T_182743 = eq(T_182742, UInt<1>(0))
      when T_182743 : idxValid[60] := UInt<1>(0)
      
      
      node T_182744 = or(T_180855, T_181041)
      node T_182745 = and(pageReplEn, T_182744)
      node T_182746 = eq(T_182745, UInt<1>(0))
      node T_182747 = eq(T_182746, UInt<1>(0))
      when T_182747 : idxValid[61] := UInt<1>(0)
      infer accessor T_182748 = idxValid[T_182499]
      T_182748 := UInt<1>(1)
      infer accessor T_182749 = idxs[T_182499]
      T_182749 := r_btb_update.bits.pc
      infer accessor T_182750 = tgts[T_182499]
      T_182750 := req.bits.addr
      infer accessor T_182751 = idxPages[T_182499]
      T_182751 := UInt<1>(0)
      infer accessor T_182752 = tgtPages[T_182499]
      T_182752 := tgtPageUpdate
      infer accessor T_182753 = useRAS[T_182499]
      T_182753 := r_btb_update.bits.isReturn
      infer accessor T_182754 = isJump[T_182499]
      T_182754 := r_btb_update.bits.isJump
      infer accessor T_182755 = brIdx[T_182499]
      T_182755 := UInt<1>(0)
      node T_182756 = and(idxPageUpdateOH, UInt<6>(21))
      node T_182757 = eq(T_182756, UInt<1>(0))
      node T_182758 = eq(T_182757, UInt<1>(0))
      wire T_182759 : UInt<1>
      T_182759 := doTgtPageRepl
      when T_182758 : T_182759 := doIdxPageRepl
      node T_182760 = shr(r_btb_update.bits.pc, 12)
      node T_182761 = shr(req.bits.addr, 12)
      wire T_182762 : UInt<27>
      T_182762 := T_182761
      when T_182758 : T_182762 := T_182760
      
      
      node T_182763 = bit(pageReplEn, 0)
      node T_182764 = and(T_182759, T_182763)
      when T_182764 : 
        infer accessor T_182765 = pages[UInt<1>(0)]
        T_182765 := T_182762
      
      
      node T_182766 = bit(pageReplEn, 2)
      node T_182767 = and(T_182759, T_182766)
      when T_182767 : 
        infer accessor T_182768 = pages[UInt<2>(2)]
        T_182768 := T_182762
      
      
      node T_182769 = bit(pageReplEn, 4)
      node T_182770 = and(T_182759, T_182769)
      when T_182770 : 
        infer accessor T_182771 = pages[UInt<3>(4)]
        T_182771 := T_182762
      wire T_182772 : UInt<1>
      T_182772 := doIdxPageRepl
      when T_182758 : T_182772 := doTgtPageRepl
      node T_182773 = shr(req.bits.addr, 12)
      node T_182774 = shr(r_btb_update.bits.pc, 12)
      wire T_182775 : UInt<27>
      T_182775 := T_182774
      when T_182758 : T_182775 := T_182773
      
      
      node T_182776 = bit(pageReplEn, 1)
      node T_182777 = and(T_182772, T_182776)
      when T_182777 : 
        infer accessor T_182778 = pages[UInt<1>(1)]
        T_182778 := T_182775
      
      
      node T_182779 = bit(pageReplEn, 3)
      node T_182780 = and(T_182772, T_182779)
      when T_182780 : 
        infer accessor T_182781 = pages[UInt<2>(3)]
        T_182781 := T_182775
      
      
      node T_182782 = bit(pageReplEn, 5)
      node T_182783 = and(T_182772, T_182782)
      when T_182783 : 
        infer accessor T_182784 = pages[UInt<3>(5)]
        T_182784 := T_182775
      when doPageRepl : 
        node T_182785 = or(pageValid, pageReplEn)
        pageValid := T_182785
    when invalidate : 
      idxValid := UInt<1>(0)
      pageValid := UInt<1>(0)
    node T_182786 = or(hits[0], hits[1])
    node T_182787 = or(T_182786, hits[2])
    node T_182788 = or(T_182787, hits[3])
    node T_182789 = or(T_182788, hits[4])
    node T_182790 = or(T_182789, hits[5])
    node T_182791 = or(T_182790, hits[6])
    node T_182792 = or(T_182791, hits[7])
    node T_182793 = or(T_182792, hits[8])
    node T_182794 = or(T_182793, hits[9])
    node T_182795 = or(T_182794, hits[10])
    node T_182796 = or(T_182795, hits[11])
    node T_182797 = or(T_182796, hits[12])
    node T_182798 = or(T_182797, hits[13])
    node T_182799 = or(T_182798, hits[14])
    node T_182800 = or(T_182799, hits[15])
    node T_182801 = or(T_182800, hits[16])
    node T_182802 = or(T_182801, hits[17])
    node T_182803 = or(T_182802, hits[18])
    node T_182804 = or(T_182803, hits[19])
    node T_182805 = or(T_182804, hits[20])
    node T_182806 = or(T_182805, hits[21])
    node T_182807 = or(T_182806, hits[22])
    node T_182808 = or(T_182807, hits[23])
    node T_182809 = or(T_182808, hits[24])
    node T_182810 = or(T_182809, hits[25])
    node T_182811 = or(T_182810, hits[26])
    node T_182812 = or(T_182811, hits[27])
    node T_182813 = or(T_182812, hits[28])
    node T_182814 = or(T_182813, hits[29])
    node T_182815 = or(T_182814, hits[30])
    node T_182816 = or(T_182815, hits[31])
    node T_182817 = or(T_182816, hits[32])
    node T_182818 = or(T_182817, hits[33])
    node T_182819 = or(T_182818, hits[34])
    node T_182820 = or(T_182819, hits[35])
    node T_182821 = or(T_182820, hits[36])
    node T_182822 = or(T_182821, hits[37])
    node T_182823 = or(T_182822, hits[38])
    node T_182824 = or(T_182823, hits[39])
    node T_182825 = or(T_182824, hits[40])
    node T_182826 = or(T_182825, hits[41])
    node T_182827 = or(T_182826, hits[42])
    node T_182828 = or(T_182827, hits[43])
    node T_182829 = or(T_182828, hits[44])
    node T_182830 = or(T_182829, hits[45])
    node T_182831 = or(T_182830, hits[46])
    node T_182832 = or(T_182831, hits[47])
    node T_182833 = or(T_182832, hits[48])
    node T_182834 = or(T_182833, hits[49])
    node T_182835 = or(T_182834, hits[50])
    node T_182836 = or(T_182835, hits[51])
    node T_182837 = or(T_182836, hits[52])
    node T_182838 = or(T_182837, hits[53])
    node T_182839 = or(T_182838, hits[54])
    node T_182840 = or(T_182839, hits[55])
    node T_182841 = or(T_182840, hits[56])
    node T_182842 = or(T_182841, hits[57])
    node T_182843 = or(T_182842, hits[58])
    node T_182844 = or(T_182843, hits[59])
    node T_182845 = or(T_182844, hits[60])
    node T_182846 = or(T_182845, hits[61])
    resp.valid := T_182846
    resp.bits.taken := resp.valid
    wire T_182847 : UInt<1>
    T_182847 := UInt<1>(0)
    when hits[0] : T_182847 := T_180858
    wire T_182848 : UInt<1>
    T_182848 := UInt<1>(0)
    when hits[1] : T_182848 := T_180861
    wire T_182849 : UInt<1>
    T_182849 := UInt<1>(0)
    when hits[2] : T_182849 := T_180864
    wire T_182850 : UInt<1>
    T_182850 := UInt<1>(0)
    when hits[3] : T_182850 := T_180867
    wire T_182851 : UInt<1>
    T_182851 := UInt<1>(0)
    when hits[4] : T_182851 := T_180870
    wire T_182852 : UInt<1>
    T_182852 := UInt<1>(0)
    when hits[5] : T_182852 := T_180873
    wire T_182853 : UInt<1>
    T_182853 := UInt<1>(0)
    when hits[6] : T_182853 := T_180876
    wire T_182854 : UInt<1>
    T_182854 := UInt<1>(0)
    when hits[7] : T_182854 := T_180879
    wire T_182855 : UInt<1>
    T_182855 := UInt<1>(0)
    when hits[8] : T_182855 := T_180882
    wire T_182856 : UInt<1>
    T_182856 := UInt<1>(0)
    when hits[9] : T_182856 := T_180885
    wire T_182857 : UInt<1>
    T_182857 := UInt<1>(0)
    when hits[10] : T_182857 := T_180888
    wire T_182858 : UInt<1>
    T_182858 := UInt<1>(0)
    when hits[11] : T_182858 := T_180891
    wire T_182859 : UInt<1>
    T_182859 := UInt<1>(0)
    when hits[12] : T_182859 := T_180894
    wire T_182860 : UInt<1>
    T_182860 := UInt<1>(0)
    when hits[13] : T_182860 := T_180897
    wire T_182861 : UInt<1>
    T_182861 := UInt<1>(0)
    when hits[14] : T_182861 := T_180900
    wire T_182862 : UInt<1>
    T_182862 := UInt<1>(0)
    when hits[15] : T_182862 := T_180903
    wire T_182863 : UInt<1>
    T_182863 := UInt<1>(0)
    when hits[16] : T_182863 := T_180906
    wire T_182864 : UInt<1>
    T_182864 := UInt<1>(0)
    when hits[17] : T_182864 := T_180909
    wire T_182865 : UInt<1>
    T_182865 := UInt<1>(0)
    when hits[18] : T_182865 := T_180912
    wire T_182866 : UInt<1>
    T_182866 := UInt<1>(0)
    when hits[19] : T_182866 := T_180915
    wire T_182867 : UInt<1>
    T_182867 := UInt<1>(0)
    when hits[20] : T_182867 := T_180918
    wire T_182868 : UInt<1>
    T_182868 := UInt<1>(0)
    when hits[21] : T_182868 := T_180921
    wire T_182869 : UInt<1>
    T_182869 := UInt<1>(0)
    when hits[22] : T_182869 := T_180924
    wire T_182870 : UInt<1>
    T_182870 := UInt<1>(0)
    when hits[23] : T_182870 := T_180927
    wire T_182871 : UInt<1>
    T_182871 := UInt<1>(0)
    when hits[24] : T_182871 := T_180930
    wire T_182872 : UInt<1>
    T_182872 := UInt<1>(0)
    when hits[25] : T_182872 := T_180933
    wire T_182873 : UInt<1>
    T_182873 := UInt<1>(0)
    when hits[26] : T_182873 := T_180936
    wire T_182874 : UInt<1>
    T_182874 := UInt<1>(0)
    when hits[27] : T_182874 := T_180939
    wire T_182875 : UInt<1>
    T_182875 := UInt<1>(0)
    when hits[28] : T_182875 := T_180942
    wire T_182876 : UInt<1>
    T_182876 := UInt<1>(0)
    when hits[29] : T_182876 := T_180945
    wire T_182877 : UInt<1>
    T_182877 := UInt<1>(0)
    when hits[30] : T_182877 := T_180948
    wire T_182878 : UInt<1>
    T_182878 := UInt<1>(0)
    when hits[31] : T_182878 := T_180951
    wire T_182879 : UInt<1>
    T_182879 := UInt<1>(0)
    when hits[32] : T_182879 := T_180954
    wire T_182880 : UInt<1>
    T_182880 := UInt<1>(0)
    when hits[33] : T_182880 := T_180957
    wire T_182881 : UInt<1>
    T_182881 := UInt<1>(0)
    when hits[34] : T_182881 := T_180960
    wire T_182882 : UInt<1>
    T_182882 := UInt<1>(0)
    when hits[35] : T_182882 := T_180963
    wire T_182883 : UInt<1>
    T_182883 := UInt<1>(0)
    when hits[36] : T_182883 := T_180966
    wire T_182884 : UInt<1>
    T_182884 := UInt<1>(0)
    when hits[37] : T_182884 := T_180969
    wire T_182885 : UInt<1>
    T_182885 := UInt<1>(0)
    when hits[38] : T_182885 := T_180972
    wire T_182886 : UInt<1>
    T_182886 := UInt<1>(0)
    when hits[39] : T_182886 := T_180975
    wire T_182887 : UInt<1>
    T_182887 := UInt<1>(0)
    when hits[40] : T_182887 := T_180978
    wire T_182888 : UInt<1>
    T_182888 := UInt<1>(0)
    when hits[41] : T_182888 := T_180981
    wire T_182889 : UInt<1>
    T_182889 := UInt<1>(0)
    when hits[42] : T_182889 := T_180984
    wire T_182890 : UInt<1>
    T_182890 := UInt<1>(0)
    when hits[43] : T_182890 := T_180987
    wire T_182891 : UInt<1>
    T_182891 := UInt<1>(0)
    when hits[44] : T_182891 := T_180990
    wire T_182892 : UInt<1>
    T_182892 := UInt<1>(0)
    when hits[45] : T_182892 := T_180993
    wire T_182893 : UInt<1>
    T_182893 := UInt<1>(0)
    when hits[46] : T_182893 := T_180996
    wire T_182894 : UInt<1>
    T_182894 := UInt<1>(0)
    when hits[47] : T_182894 := T_180999
    wire T_182895 : UInt<1>
    T_182895 := UInt<1>(0)
    when hits[48] : T_182895 := T_181002
    wire T_182896 : UInt<1>
    T_182896 := UInt<1>(0)
    when hits[49] : T_182896 := T_181005
    wire T_182897 : UInt<1>
    T_182897 := UInt<1>(0)
    when hits[50] : T_182897 := T_181008
    wire T_182898 : UInt<1>
    T_182898 := UInt<1>(0)
    when hits[51] : T_182898 := T_181011
    wire T_182899 : UInt<1>
    T_182899 := UInt<1>(0)
    when hits[52] : T_182899 := T_181014
    wire T_182900 : UInt<1>
    T_182900 := UInt<1>(0)
    when hits[53] : T_182900 := T_181017
    wire T_182901 : UInt<1>
    T_182901 := UInt<1>(0)
    when hits[54] : T_182901 := T_181020
    wire T_182902 : UInt<1>
    T_182902 := UInt<1>(0)
    when hits[55] : T_182902 := T_181023
    wire T_182903 : UInt<1>
    T_182903 := UInt<1>(0)
    when hits[56] : T_182903 := T_181026
    wire T_182904 : UInt<1>
    T_182904 := UInt<1>(0)
    when hits[57] : T_182904 := T_181029
    wire T_182905 : UInt<1>
    T_182905 := UInt<1>(0)
    when hits[58] : T_182905 := T_181032
    wire T_182906 : UInt<1>
    T_182906 := UInt<1>(0)
    when hits[59] : T_182906 := T_181035
    wire T_182907 : UInt<1>
    T_182907 := UInt<1>(0)
    when hits[60] : T_182907 := T_181038
    wire T_182908 : UInt<1>
    T_182908 := UInt<1>(0)
    when hits[61] : T_182908 := T_181041
    node T_182909 = or(T_182847, T_182848)
    node T_182910 = or(T_182909, T_182849)
    node T_182911 = or(T_182910, T_182850)
    node T_182912 = or(T_182911, T_182851)
    node T_182913 = or(T_182912, T_182852)
    node T_182914 = or(T_182913, T_182853)
    node T_182915 = or(T_182914, T_182854)
    node T_182916 = or(T_182915, T_182855)
    node T_182917 = or(T_182916, T_182856)
    node T_182918 = or(T_182917, T_182857)
    node T_182919 = or(T_182918, T_182858)
    node T_182920 = or(T_182919, T_182859)
    node T_182921 = or(T_182920, T_182860)
    node T_182922 = or(T_182921, T_182861)
    node T_182923 = or(T_182922, T_182862)
    node T_182924 = or(T_182923, T_182863)
    node T_182925 = or(T_182924, T_182864)
    node T_182926 = or(T_182925, T_182865)
    node T_182927 = or(T_182926, T_182866)
    node T_182928 = or(T_182927, T_182867)
    node T_182929 = or(T_182928, T_182868)
    node T_182930 = or(T_182929, T_182869)
    node T_182931 = or(T_182930, T_182870)
    node T_182932 = or(T_182931, T_182871)
    node T_182933 = or(T_182932, T_182872)
    node T_182934 = or(T_182933, T_182873)
    node T_182935 = or(T_182934, T_182874)
    node T_182936 = or(T_182935, T_182875)
    node T_182937 = or(T_182936, T_182876)
    node T_182938 = or(T_182937, T_182877)
    node T_182939 = or(T_182938, T_182878)
    node T_182940 = or(T_182939, T_182879)
    node T_182941 = or(T_182940, T_182880)
    node T_182942 = or(T_182941, T_182881)
    node T_182943 = or(T_182942, T_182882)
    node T_182944 = or(T_182943, T_182883)
    node T_182945 = or(T_182944, T_182884)
    node T_182946 = or(T_182945, T_182885)
    node T_182947 = or(T_182946, T_182886)
    node T_182948 = or(T_182947, T_182887)
    node T_182949 = or(T_182948, T_182888)
    node T_182950 = or(T_182949, T_182889)
    node T_182951 = or(T_182950, T_182890)
    node T_182952 = or(T_182951, T_182891)
    node T_182953 = or(T_182952, T_182892)
    node T_182954 = or(T_182953, T_182893)
    node T_182955 = or(T_182954, T_182894)
    node T_182956 = or(T_182955, T_182895)
    node T_182957 = or(T_182956, T_182896)
    node T_182958 = or(T_182957, T_182897)
    node T_182959 = or(T_182958, T_182898)
    node T_182960 = or(T_182959, T_182899)
    node T_182961 = or(T_182960, T_182900)
    node T_182962 = or(T_182961, T_182901)
    node T_182963 = or(T_182962, T_182902)
    node T_182964 = or(T_182963, T_182903)
    node T_182965 = or(T_182964, T_182904)
    node T_182966 = or(T_182965, T_182905)
    node T_182967 = or(T_182966, T_182906)
    node T_182968 = or(T_182967, T_182907)
    node T_182969 = or(T_182968, T_182908)
    wire T_182970 : UInt<6>
    node T_182971 = bits(T_182969, 5, 0)
    T_182970 := T_182971
    node T_182972 = bit(T_182970, 0)
    node T_182973 = bit(T_182970, 1)
    node T_182974 = bit(T_182970, 2)
    node T_182975 = bit(T_182970, 3)
    node T_182976 = bit(T_182970, 4)
    node T_182977 = bit(T_182970, 5)
    infer accessor T_182978 = pages[UInt<1>(0)]
    infer accessor T_182979 = pages[UInt<1>(1)]
    infer accessor T_182980 = pages[UInt<2>(2)]
    infer accessor T_182981 = pages[UInt<2>(3)]
    infer accessor T_182982 = pages[UInt<3>(4)]
    infer accessor T_182983 = pages[UInt<3>(5)]
    wire T_182984 : UInt<1>
    T_182984 := UInt<1>(0)
    when T_182972 : T_182984 := T_182978
    wire T_182985 : UInt<1>
    T_182985 := UInt<1>(0)
    when T_182973 : T_182985 := T_182979
    wire T_182986 : UInt<1>
    T_182986 := UInt<1>(0)
    when T_182974 : T_182986 := T_182980
    wire T_182987 : UInt<1>
    T_182987 := UInt<1>(0)
    when T_182975 : T_182987 := T_182981
    wire T_182988 : UInt<1>
    T_182988 := UInt<1>(0)
    when T_182976 : T_182988 := T_182982
    wire T_182989 : UInt<1>
    T_182989 := UInt<1>(0)
    when T_182977 : T_182989 := T_182983
    node T_182990 = or(T_182984, T_182985)
    node T_182991 = or(T_182990, T_182986)
    node T_182992 = or(T_182991, T_182987)
    node T_182993 = or(T_182992, T_182988)
    node T_182994 = or(T_182993, T_182989)
    wire T_182995 : UInt<27>
    node T_182996 = bits(T_182994, 26, 0)
    T_182995 := T_182996
    infer accessor T_182997 = tgts[UInt<1>(0)]
    infer accessor T_182998 = tgts[UInt<1>(1)]
    infer accessor T_182999 = tgts[UInt<2>(2)]
    infer accessor T_183000 = tgts[UInt<2>(3)]
    infer accessor T_183001 = tgts[UInt<3>(4)]
    infer accessor T_183002 = tgts[UInt<3>(5)]
    infer accessor T_183003 = tgts[UInt<3>(6)]
    infer accessor T_183004 = tgts[UInt<3>(7)]
    infer accessor T_183005 = tgts[UInt<4>(8)]
    infer accessor T_183006 = tgts[UInt<4>(9)]
    infer accessor T_183007 = tgts[UInt<4>(10)]
    infer accessor T_183008 = tgts[UInt<4>(11)]
    infer accessor T_183009 = tgts[UInt<4>(12)]
    infer accessor T_183010 = tgts[UInt<4>(13)]
    infer accessor T_183011 = tgts[UInt<4>(14)]
    infer accessor T_183012 = tgts[UInt<4>(15)]
    infer accessor T_183013 = tgts[UInt<5>(16)]
    infer accessor T_183014 = tgts[UInt<5>(17)]
    infer accessor T_183015 = tgts[UInt<5>(18)]
    infer accessor T_183016 = tgts[UInt<5>(19)]
    infer accessor T_183017 = tgts[UInt<5>(20)]
    infer accessor T_183018 = tgts[UInt<5>(21)]
    infer accessor T_183019 = tgts[UInt<5>(22)]
    infer accessor T_183020 = tgts[UInt<5>(23)]
    infer accessor T_183021 = tgts[UInt<5>(24)]
    infer accessor T_183022 = tgts[UInt<5>(25)]
    infer accessor T_183023 = tgts[UInt<5>(26)]
    infer accessor T_183024 = tgts[UInt<5>(27)]
    infer accessor T_183025 = tgts[UInt<5>(28)]
    infer accessor T_183026 = tgts[UInt<5>(29)]
    infer accessor T_183027 = tgts[UInt<5>(30)]
    infer accessor T_183028 = tgts[UInt<5>(31)]
    infer accessor T_183029 = tgts[UInt<6>(32)]
    infer accessor T_183030 = tgts[UInt<6>(33)]
    infer accessor T_183031 = tgts[UInt<6>(34)]
    infer accessor T_183032 = tgts[UInt<6>(35)]
    infer accessor T_183033 = tgts[UInt<6>(36)]
    infer accessor T_183034 = tgts[UInt<6>(37)]
    infer accessor T_183035 = tgts[UInt<6>(38)]
    infer accessor T_183036 = tgts[UInt<6>(39)]
    infer accessor T_183037 = tgts[UInt<6>(40)]
    infer accessor T_183038 = tgts[UInt<6>(41)]
    infer accessor T_183039 = tgts[UInt<6>(42)]
    infer accessor T_183040 = tgts[UInt<6>(43)]
    infer accessor T_183041 = tgts[UInt<6>(44)]
    infer accessor T_183042 = tgts[UInt<6>(45)]
    infer accessor T_183043 = tgts[UInt<6>(46)]
    infer accessor T_183044 = tgts[UInt<6>(47)]
    infer accessor T_183045 = tgts[UInt<6>(48)]
    infer accessor T_183046 = tgts[UInt<6>(49)]
    infer accessor T_183047 = tgts[UInt<6>(50)]
    infer accessor T_183048 = tgts[UInt<6>(51)]
    infer accessor T_183049 = tgts[UInt<6>(52)]
    infer accessor T_183050 = tgts[UInt<6>(53)]
    infer accessor T_183051 = tgts[UInt<6>(54)]
    infer accessor T_183052 = tgts[UInt<6>(55)]
    infer accessor T_183053 = tgts[UInt<6>(56)]
    infer accessor T_183054 = tgts[UInt<6>(57)]
    infer accessor T_183055 = tgts[UInt<6>(58)]
    infer accessor T_183056 = tgts[UInt<6>(59)]
    infer accessor T_183057 = tgts[UInt<6>(60)]
    infer accessor T_183058 = tgts[UInt<6>(61)]
    wire T_183059 : UInt<1>
    T_183059 := UInt<1>(0)
    when hits[0] : T_183059 := T_182997
    wire T_183060 : UInt<1>
    T_183060 := UInt<1>(0)
    when hits[1] : T_183060 := T_182998
    wire T_183061 : UInt<1>
    T_183061 := UInt<1>(0)
    when hits[2] : T_183061 := T_182999
    wire T_183062 : UInt<1>
    T_183062 := UInt<1>(0)
    when hits[3] : T_183062 := T_183000
    wire T_183063 : UInt<1>
    T_183063 := UInt<1>(0)
    when hits[4] : T_183063 := T_183001
    wire T_183064 : UInt<1>
    T_183064 := UInt<1>(0)
    when hits[5] : T_183064 := T_183002
    wire T_183065 : UInt<1>
    T_183065 := UInt<1>(0)
    when hits[6] : T_183065 := T_183003
    wire T_183066 : UInt<1>
    T_183066 := UInt<1>(0)
    when hits[7] : T_183066 := T_183004
    wire T_183067 : UInt<1>
    T_183067 := UInt<1>(0)
    when hits[8] : T_183067 := T_183005
    wire T_183068 : UInt<1>
    T_183068 := UInt<1>(0)
    when hits[9] : T_183068 := T_183006
    wire T_183069 : UInt<1>
    T_183069 := UInt<1>(0)
    when hits[10] : T_183069 := T_183007
    wire T_183070 : UInt<1>
    T_183070 := UInt<1>(0)
    when hits[11] : T_183070 := T_183008
    wire T_183071 : UInt<1>
    T_183071 := UInt<1>(0)
    when hits[12] : T_183071 := T_183009
    wire T_183072 : UInt<1>
    T_183072 := UInt<1>(0)
    when hits[13] : T_183072 := T_183010
    wire T_183073 : UInt<1>
    T_183073 := UInt<1>(0)
    when hits[14] : T_183073 := T_183011
    wire T_183074 : UInt<1>
    T_183074 := UInt<1>(0)
    when hits[15] : T_183074 := T_183012
    wire T_183075 : UInt<1>
    T_183075 := UInt<1>(0)
    when hits[16] : T_183075 := T_183013
    wire T_183076 : UInt<1>
    T_183076 := UInt<1>(0)
    when hits[17] : T_183076 := T_183014
    wire T_183077 : UInt<1>
    T_183077 := UInt<1>(0)
    when hits[18] : T_183077 := T_183015
    wire T_183078 : UInt<1>
    T_183078 := UInt<1>(0)
    when hits[19] : T_183078 := T_183016
    wire T_183079 : UInt<1>
    T_183079 := UInt<1>(0)
    when hits[20] : T_183079 := T_183017
    wire T_183080 : UInt<1>
    T_183080 := UInt<1>(0)
    when hits[21] : T_183080 := T_183018
    wire T_183081 : UInt<1>
    T_183081 := UInt<1>(0)
    when hits[22] : T_183081 := T_183019
    wire T_183082 : UInt<1>
    T_183082 := UInt<1>(0)
    when hits[23] : T_183082 := T_183020
    wire T_183083 : UInt<1>
    T_183083 := UInt<1>(0)
    when hits[24] : T_183083 := T_183021
    wire T_183084 : UInt<1>
    T_183084 := UInt<1>(0)
    when hits[25] : T_183084 := T_183022
    wire T_183085 : UInt<1>
    T_183085 := UInt<1>(0)
    when hits[26] : T_183085 := T_183023
    wire T_183086 : UInt<1>
    T_183086 := UInt<1>(0)
    when hits[27] : T_183086 := T_183024
    wire T_183087 : UInt<1>
    T_183087 := UInt<1>(0)
    when hits[28] : T_183087 := T_183025
    wire T_183088 : UInt<1>
    T_183088 := UInt<1>(0)
    when hits[29] : T_183088 := T_183026
    wire T_183089 : UInt<1>
    T_183089 := UInt<1>(0)
    when hits[30] : T_183089 := T_183027
    wire T_183090 : UInt<1>
    T_183090 := UInt<1>(0)
    when hits[31] : T_183090 := T_183028
    wire T_183091 : UInt<1>
    T_183091 := UInt<1>(0)
    when hits[32] : T_183091 := T_183029
    wire T_183092 : UInt<1>
    T_183092 := UInt<1>(0)
    when hits[33] : T_183092 := T_183030
    wire T_183093 : UInt<1>
    T_183093 := UInt<1>(0)
    when hits[34] : T_183093 := T_183031
    wire T_183094 : UInt<1>
    T_183094 := UInt<1>(0)
    when hits[35] : T_183094 := T_183032
    wire T_183095 : UInt<1>
    T_183095 := UInt<1>(0)
    when hits[36] : T_183095 := T_183033
    wire T_183096 : UInt<1>
    T_183096 := UInt<1>(0)
    when hits[37] : T_183096 := T_183034
    wire T_183097 : UInt<1>
    T_183097 := UInt<1>(0)
    when hits[38] : T_183097 := T_183035
    wire T_183098 : UInt<1>
    T_183098 := UInt<1>(0)
    when hits[39] : T_183098 := T_183036
    wire T_183099 : UInt<1>
    T_183099 := UInt<1>(0)
    when hits[40] : T_183099 := T_183037
    wire T_183100 : UInt<1>
    T_183100 := UInt<1>(0)
    when hits[41] : T_183100 := T_183038
    wire T_183101 : UInt<1>
    T_183101 := UInt<1>(0)
    when hits[42] : T_183101 := T_183039
    wire T_183102 : UInt<1>
    T_183102 := UInt<1>(0)
    when hits[43] : T_183102 := T_183040
    wire T_183103 : UInt<1>
    T_183103 := UInt<1>(0)
    when hits[44] : T_183103 := T_183041
    wire T_183104 : UInt<1>
    T_183104 := UInt<1>(0)
    when hits[45] : T_183104 := T_183042
    wire T_183105 : UInt<1>
    T_183105 := UInt<1>(0)
    when hits[46] : T_183105 := T_183043
    wire T_183106 : UInt<1>
    T_183106 := UInt<1>(0)
    when hits[47] : T_183106 := T_183044
    wire T_183107 : UInt<1>
    T_183107 := UInt<1>(0)
    when hits[48] : T_183107 := T_183045
    wire T_183108 : UInt<1>
    T_183108 := UInt<1>(0)
    when hits[49] : T_183108 := T_183046
    wire T_183109 : UInt<1>
    T_183109 := UInt<1>(0)
    when hits[50] : T_183109 := T_183047
    wire T_183110 : UInt<1>
    T_183110 := UInt<1>(0)
    when hits[51] : T_183110 := T_183048
    wire T_183111 : UInt<1>
    T_183111 := UInt<1>(0)
    when hits[52] : T_183111 := T_183049
    wire T_183112 : UInt<1>
    T_183112 := UInt<1>(0)
    when hits[53] : T_183112 := T_183050
    wire T_183113 : UInt<1>
    T_183113 := UInt<1>(0)
    when hits[54] : T_183113 := T_183051
    wire T_183114 : UInt<1>
    T_183114 := UInt<1>(0)
    when hits[55] : T_183114 := T_183052
    wire T_183115 : UInt<1>
    T_183115 := UInt<1>(0)
    when hits[56] : T_183115 := T_183053
    wire T_183116 : UInt<1>
    T_183116 := UInt<1>(0)
    when hits[57] : T_183116 := T_183054
    wire T_183117 : UInt<1>
    T_183117 := UInt<1>(0)
    when hits[58] : T_183117 := T_183055
    wire T_183118 : UInt<1>
    T_183118 := UInt<1>(0)
    when hits[59] : T_183118 := T_183056
    wire T_183119 : UInt<1>
    T_183119 := UInt<1>(0)
    when hits[60] : T_183119 := T_183057
    wire T_183120 : UInt<1>
    T_183120 := UInt<1>(0)
    when hits[61] : T_183120 := T_183058
    node T_183121 = or(T_183059, T_183060)
    node T_183122 = or(T_183121, T_183061)
    node T_183123 = or(T_183122, T_183062)
    node T_183124 = or(T_183123, T_183063)
    node T_183125 = or(T_183124, T_183064)
    node T_183126 = or(T_183125, T_183065)
    node T_183127 = or(T_183126, T_183066)
    node T_183128 = or(T_183127, T_183067)
    node T_183129 = or(T_183128, T_183068)
    node T_183130 = or(T_183129, T_183069)
    node T_183131 = or(T_183130, T_183070)
    node T_183132 = or(T_183131, T_183071)
    node T_183133 = or(T_183132, T_183072)
    node T_183134 = or(T_183133, T_183073)
    node T_183135 = or(T_183134, T_183074)
    node T_183136 = or(T_183135, T_183075)
    node T_183137 = or(T_183136, T_183076)
    node T_183138 = or(T_183137, T_183077)
    node T_183139 = or(T_183138, T_183078)
    node T_183140 = or(T_183139, T_183079)
    node T_183141 = or(T_183140, T_183080)
    node T_183142 = or(T_183141, T_183081)
    node T_183143 = or(T_183142, T_183082)
    node T_183144 = or(T_183143, T_183083)
    node T_183145 = or(T_183144, T_183084)
    node T_183146 = or(T_183145, T_183085)
    node T_183147 = or(T_183146, T_183086)
    node T_183148 = or(T_183147, T_183087)
    node T_183149 = or(T_183148, T_183088)
    node T_183150 = or(T_183149, T_183089)
    node T_183151 = or(T_183150, T_183090)
    node T_183152 = or(T_183151, T_183091)
    node T_183153 = or(T_183152, T_183092)
    node T_183154 = or(T_183153, T_183093)
    node T_183155 = or(T_183154, T_183094)
    node T_183156 = or(T_183155, T_183095)
    node T_183157 = or(T_183156, T_183096)
    node T_183158 = or(T_183157, T_183097)
    node T_183159 = or(T_183158, T_183098)
    node T_183160 = or(T_183159, T_183099)
    node T_183161 = or(T_183160, T_183100)
    node T_183162 = or(T_183161, T_183101)
    node T_183163 = or(T_183162, T_183102)
    node T_183164 = or(T_183163, T_183103)
    node T_183165 = or(T_183164, T_183104)
    node T_183166 = or(T_183165, T_183105)
    node T_183167 = or(T_183166, T_183106)
    node T_183168 = or(T_183167, T_183107)
    node T_183169 = or(T_183168, T_183108)
    node T_183170 = or(T_183169, T_183109)
    node T_183171 = or(T_183170, T_183110)
    node T_183172 = or(T_183171, T_183111)
    node T_183173 = or(T_183172, T_183112)
    node T_183174 = or(T_183173, T_183113)
    node T_183175 = or(T_183174, T_183114)
    node T_183176 = or(T_183175, T_183115)
    node T_183177 = or(T_183176, T_183116)
    node T_183178 = or(T_183177, T_183117)
    node T_183179 = or(T_183178, T_183118)
    node T_183180 = or(T_183179, T_183119)
    node T_183181 = or(T_183180, T_183120)
    wire T_183182 : UInt<12>
    node T_183183 = bits(T_183181, 11, 0)
    T_183182 := T_183183
    node T_183184 = cat(T_182995, T_183182)
    resp.bits.target := T_183184
    node T_183185 = cat(hits[60], hits[59])
    node T_183186 = cat(hits[61], T_183185)
    node T_183187 = cat(hits[58], hits[57])
    node T_183188 = cat(hits[56], hits[55])
    node T_183189 = cat(T_183187, T_183188)
    node T_183190 = cat(T_183186, T_183189)
    node T_183191 = cat(hits[54], hits[53])
    node T_183192 = cat(hits[52], hits[51])
    node T_183193 = cat(T_183191, T_183192)
    node T_183194 = cat(hits[50], hits[49])
    node T_183195 = cat(hits[48], hits[47])
    node T_183196 = cat(T_183194, T_183195)
    node T_183197 = cat(T_183193, T_183196)
    node T_183198 = cat(T_183190, T_183197)
    node T_183199 = cat(hits[46], hits[45])
    node T_183200 = cat(hits[44], hits[43])
    node T_183201 = cat(T_183199, T_183200)
    node T_183202 = cat(hits[42], hits[41])
    node T_183203 = cat(hits[40], hits[39])
    node T_183204 = cat(T_183202, T_183203)
    node T_183205 = cat(T_183201, T_183204)
    node T_183206 = cat(hits[38], hits[37])
    node T_183207 = cat(hits[36], hits[35])
    node T_183208 = cat(T_183206, T_183207)
    node T_183209 = cat(hits[34], hits[33])
    node T_183210 = cat(hits[32], hits[31])
    node T_183211 = cat(T_183209, T_183210)
    node T_183212 = cat(T_183208, T_183211)
    node T_183213 = cat(T_183205, T_183212)
    node T_183214 = cat(T_183198, T_183213)
    node T_183215 = cat(hits[29], hits[28])
    node T_183216 = cat(hits[30], T_183215)
    node T_183217 = cat(hits[27], hits[26])
    node T_183218 = cat(hits[25], hits[24])
    node T_183219 = cat(T_183217, T_183218)
    node T_183220 = cat(T_183216, T_183219)
    node T_183221 = cat(hits[23], hits[22])
    node T_183222 = cat(hits[21], hits[20])
    node T_183223 = cat(T_183221, T_183222)
    node T_183224 = cat(hits[19], hits[18])
    node T_183225 = cat(hits[17], hits[16])
    node T_183226 = cat(T_183224, T_183225)
    node T_183227 = cat(T_183223, T_183226)
    node T_183228 = cat(T_183220, T_183227)
    node T_183229 = cat(hits[15], hits[14])
    node T_183230 = cat(hits[13], hits[12])
    node T_183231 = cat(T_183229, T_183230)
    node T_183232 = cat(hits[11], hits[10])
    node T_183233 = cat(hits[9], hits[8])
    node T_183234 = cat(T_183232, T_183233)
    node T_183235 = cat(T_183231, T_183234)
    node T_183236 = cat(hits[7], hits[6])
    node T_183237 = cat(hits[5], hits[4])
    node T_183238 = cat(T_183236, T_183237)
    node T_183239 = cat(hits[3], hits[2])
    node T_183240 = cat(hits[1], hits[0])
    node T_183241 = cat(T_183239, T_183240)
    node T_183242 = cat(T_183238, T_183241)
    node T_183243 = cat(T_183235, T_183242)
    node T_183244 = cat(T_183228, T_183243)
    node T_183245 = cat(T_183214, T_183244)
    node T_183246 = bits(T_183245, 61, 32)
    node T_183247 = bits(T_183245, 31, 0)
    node T_183248 = eq(T_183246, UInt<1>(0))
    node T_183249 = eq(T_183248, UInt<1>(0))
    node T_183250 = or(T_183246, T_183247)
    node T_183251 = bits(T_183250, 31, 16)
    node T_183252 = bits(T_183250, 15, 0)
    node T_183253 = eq(T_183251, UInt<1>(0))
    node T_183254 = eq(T_183253, UInt<1>(0))
    node T_183255 = or(T_183251, T_183252)
    node T_183256 = bits(T_183255, 15, 8)
    node T_183257 = bits(T_183255, 7, 0)
    node T_183258 = eq(T_183256, UInt<1>(0))
    node T_183259 = eq(T_183258, UInt<1>(0))
    node T_183260 = or(T_183256, T_183257)
    node T_183261 = bits(T_183260, 7, 4)
    node T_183262 = bits(T_183260, 3, 0)
    node T_183263 = eq(T_183261, UInt<1>(0))
    node T_183264 = eq(T_183263, UInt<1>(0))
    node T_183265 = or(T_183261, T_183262)
    node T_183266 = bits(T_183265, 3, 2)
    node T_183267 = bits(T_183265, 1, 0)
    node T_183268 = eq(T_183266, UInt<1>(0))
    node T_183269 = eq(T_183268, UInt<1>(0))
    node T_183270 = or(T_183266, T_183267)
    node T_183271 = bit(T_183270, 1)
    node T_183272 = cat(T_183269, T_183271)
    node T_183273 = cat(T_183264, T_183272)
    node T_183274 = cat(T_183259, T_183273)
    node T_183275 = cat(T_183254, T_183274)
    node T_183276 = cat(T_183249, T_183275)
    resp.bits.entry := T_183276
    infer accessor T_183277 = brIdx[resp.bits.entry]
    resp.bits.bridx := T_183277
    resp.bits.mask := UInt<1>(1)
    cmem T_183278 : UInt<2>[128], clk
    reg T_183279 : UInt<7>, clk, reset
    wire T_183280 : UInt<1>
    T_183280 := UInt<1>(0)
    when hits[0] : T_183280 := isJump[0]
    wire T_183281 : UInt<1>
    T_183281 := UInt<1>(0)
    when hits[1] : T_183281 := isJump[1]
    wire T_183282 : UInt<1>
    T_183282 := UInt<1>(0)
    when hits[2] : T_183282 := isJump[2]
    wire T_183283 : UInt<1>
    T_183283 := UInt<1>(0)
    when hits[3] : T_183283 := isJump[3]
    wire T_183284 : UInt<1>
    T_183284 := UInt<1>(0)
    when hits[4] : T_183284 := isJump[4]
    wire T_183285 : UInt<1>
    T_183285 := UInt<1>(0)
    when hits[5] : T_183285 := isJump[5]
    wire T_183286 : UInt<1>
    T_183286 := UInt<1>(0)
    when hits[6] : T_183286 := isJump[6]
    wire T_183287 : UInt<1>
    T_183287 := UInt<1>(0)
    when hits[7] : T_183287 := isJump[7]
    wire T_183288 : UInt<1>
    T_183288 := UInt<1>(0)
    when hits[8] : T_183288 := isJump[8]
    wire T_183289 : UInt<1>
    T_183289 := UInt<1>(0)
    when hits[9] : T_183289 := isJump[9]
    wire T_183290 : UInt<1>
    T_183290 := UInt<1>(0)
    when hits[10] : T_183290 := isJump[10]
    wire T_183291 : UInt<1>
    T_183291 := UInt<1>(0)
    when hits[11] : T_183291 := isJump[11]
    wire T_183292 : UInt<1>
    T_183292 := UInt<1>(0)
    when hits[12] : T_183292 := isJump[12]
    wire T_183293 : UInt<1>
    T_183293 := UInt<1>(0)
    when hits[13] : T_183293 := isJump[13]
    wire T_183294 : UInt<1>
    T_183294 := UInt<1>(0)
    when hits[14] : T_183294 := isJump[14]
    wire T_183295 : UInt<1>
    T_183295 := UInt<1>(0)
    when hits[15] : T_183295 := isJump[15]
    wire T_183296 : UInt<1>
    T_183296 := UInt<1>(0)
    when hits[16] : T_183296 := isJump[16]
    wire T_183297 : UInt<1>
    T_183297 := UInt<1>(0)
    when hits[17] : T_183297 := isJump[17]
    wire T_183298 : UInt<1>
    T_183298 := UInt<1>(0)
    when hits[18] : T_183298 := isJump[18]
    wire T_183299 : UInt<1>
    T_183299 := UInt<1>(0)
    when hits[19] : T_183299 := isJump[19]
    wire T_183300 : UInt<1>
    T_183300 := UInt<1>(0)
    when hits[20] : T_183300 := isJump[20]
    wire T_183301 : UInt<1>
    T_183301 := UInt<1>(0)
    when hits[21] : T_183301 := isJump[21]
    wire T_183302 : UInt<1>
    T_183302 := UInt<1>(0)
    when hits[22] : T_183302 := isJump[22]
    wire T_183303 : UInt<1>
    T_183303 := UInt<1>(0)
    when hits[23] : T_183303 := isJump[23]
    wire T_183304 : UInt<1>
    T_183304 := UInt<1>(0)
    when hits[24] : T_183304 := isJump[24]
    wire T_183305 : UInt<1>
    T_183305 := UInt<1>(0)
    when hits[25] : T_183305 := isJump[25]
    wire T_183306 : UInt<1>
    T_183306 := UInt<1>(0)
    when hits[26] : T_183306 := isJump[26]
    wire T_183307 : UInt<1>
    T_183307 := UInt<1>(0)
    when hits[27] : T_183307 := isJump[27]
    wire T_183308 : UInt<1>
    T_183308 := UInt<1>(0)
    when hits[28] : T_183308 := isJump[28]
    wire T_183309 : UInt<1>
    T_183309 := UInt<1>(0)
    when hits[29] : T_183309 := isJump[29]
    wire T_183310 : UInt<1>
    T_183310 := UInt<1>(0)
    when hits[30] : T_183310 := isJump[30]
    wire T_183311 : UInt<1>
    T_183311 := UInt<1>(0)
    when hits[31] : T_183311 := isJump[31]
    wire T_183312 : UInt<1>
    T_183312 := UInt<1>(0)
    when hits[32] : T_183312 := isJump[32]
    wire T_183313 : UInt<1>
    T_183313 := UInt<1>(0)
    when hits[33] : T_183313 := isJump[33]
    wire T_183314 : UInt<1>
    T_183314 := UInt<1>(0)
    when hits[34] : T_183314 := isJump[34]
    wire T_183315 : UInt<1>
    T_183315 := UInt<1>(0)
    when hits[35] : T_183315 := isJump[35]
    wire T_183316 : UInt<1>
    T_183316 := UInt<1>(0)
    when hits[36] : T_183316 := isJump[36]
    wire T_183317 : UInt<1>
    T_183317 := UInt<1>(0)
    when hits[37] : T_183317 := isJump[37]
    wire T_183318 : UInt<1>
    T_183318 := UInt<1>(0)
    when hits[38] : T_183318 := isJump[38]
    wire T_183319 : UInt<1>
    T_183319 := UInt<1>(0)
    when hits[39] : T_183319 := isJump[39]
    wire T_183320 : UInt<1>
    T_183320 := UInt<1>(0)
    when hits[40] : T_183320 := isJump[40]
    wire T_183321 : UInt<1>
    T_183321 := UInt<1>(0)
    when hits[41] : T_183321 := isJump[41]
    wire T_183322 : UInt<1>
    T_183322 := UInt<1>(0)
    when hits[42] : T_183322 := isJump[42]
    wire T_183323 : UInt<1>
    T_183323 := UInt<1>(0)
    when hits[43] : T_183323 := isJump[43]
    wire T_183324 : UInt<1>
    T_183324 := UInt<1>(0)
    when hits[44] : T_183324 := isJump[44]
    wire T_183325 : UInt<1>
    T_183325 := UInt<1>(0)
    when hits[45] : T_183325 := isJump[45]
    wire T_183326 : UInt<1>
    T_183326 := UInt<1>(0)
    when hits[46] : T_183326 := isJump[46]
    wire T_183327 : UInt<1>
    T_183327 := UInt<1>(0)
    when hits[47] : T_183327 := isJump[47]
    wire T_183328 : UInt<1>
    T_183328 := UInt<1>(0)
    when hits[48] : T_183328 := isJump[48]
    wire T_183329 : UInt<1>
    T_183329 := UInt<1>(0)
    when hits[49] : T_183329 := isJump[49]
    wire T_183330 : UInt<1>
    T_183330 := UInt<1>(0)
    when hits[50] : T_183330 := isJump[50]
    wire T_183331 : UInt<1>
    T_183331 := UInt<1>(0)
    when hits[51] : T_183331 := isJump[51]
    wire T_183332 : UInt<1>
    T_183332 := UInt<1>(0)
    when hits[52] : T_183332 := isJump[52]
    wire T_183333 : UInt<1>
    T_183333 := UInt<1>(0)
    when hits[53] : T_183333 := isJump[53]
    wire T_183334 : UInt<1>
    T_183334 := UInt<1>(0)
    when hits[54] : T_183334 := isJump[54]
    wire T_183335 : UInt<1>
    T_183335 := UInt<1>(0)
    when hits[55] : T_183335 := isJump[55]
    wire T_183336 : UInt<1>
    T_183336 := UInt<1>(0)
    when hits[56] : T_183336 := isJump[56]
    wire T_183337 : UInt<1>
    T_183337 := UInt<1>(0)
    when hits[57] : T_183337 := isJump[57]
    wire T_183338 : UInt<1>
    T_183338 := UInt<1>(0)
    when hits[58] : T_183338 := isJump[58]
    wire T_183339 : UInt<1>
    T_183339 := UInt<1>(0)
    when hits[59] : T_183339 := isJump[59]
    wire T_183340 : UInt<1>
    T_183340 := UInt<1>(0)
    when hits[60] : T_183340 := isJump[60]
    wire T_183341 : UInt<1>
    T_183341 := UInt<1>(0)
    when hits[61] : T_183341 := isJump[61]
    node T_183342 = or(T_183280, T_183281)
    node T_183343 = or(T_183342, T_183282)
    node T_183344 = or(T_183343, T_183283)
    node T_183345 = or(T_183344, T_183284)
    node T_183346 = or(T_183345, T_183285)
    node T_183347 = or(T_183346, T_183286)
    node T_183348 = or(T_183347, T_183287)
    node T_183349 = or(T_183348, T_183288)
    node T_183350 = or(T_183349, T_183289)
    node T_183351 = or(T_183350, T_183290)
    node T_183352 = or(T_183351, T_183291)
    node T_183353 = or(T_183352, T_183292)
    node T_183354 = or(T_183353, T_183293)
    node T_183355 = or(T_183354, T_183294)
    node T_183356 = or(T_183355, T_183295)
    node T_183357 = or(T_183356, T_183296)
    node T_183358 = or(T_183357, T_183297)
    node T_183359 = or(T_183358, T_183298)
    node T_183360 = or(T_183359, T_183299)
    node T_183361 = or(T_183360, T_183300)
    node T_183362 = or(T_183361, T_183301)
    node T_183363 = or(T_183362, T_183302)
    node T_183364 = or(T_183363, T_183303)
    node T_183365 = or(T_183364, T_183304)
    node T_183366 = or(T_183365, T_183305)
    node T_183367 = or(T_183366, T_183306)
    node T_183368 = or(T_183367, T_183307)
    node T_183369 = or(T_183368, T_183308)
    node T_183370 = or(T_183369, T_183309)
    node T_183371 = or(T_183370, T_183310)
    node T_183372 = or(T_183371, T_183311)
    node T_183373 = or(T_183372, T_183312)
    node T_183374 = or(T_183373, T_183313)
    node T_183375 = or(T_183374, T_183314)
    node T_183376 = or(T_183375, T_183315)
    node T_183377 = or(T_183376, T_183316)
    node T_183378 = or(T_183377, T_183317)
    node T_183379 = or(T_183378, T_183318)
    node T_183380 = or(T_183379, T_183319)
    node T_183381 = or(T_183380, T_183320)
    node T_183382 = or(T_183381, T_183321)
    node T_183383 = or(T_183382, T_183322)
    node T_183384 = or(T_183383, T_183323)
    node T_183385 = or(T_183384, T_183324)
    node T_183386 = or(T_183385, T_183325)
    node T_183387 = or(T_183386, T_183326)
    node T_183388 = or(T_183387, T_183327)
    node T_183389 = or(T_183388, T_183328)
    node T_183390 = or(T_183389, T_183329)
    node T_183391 = or(T_183390, T_183330)
    node T_183392 = or(T_183391, T_183331)
    node T_183393 = or(T_183392, T_183332)
    node T_183394 = or(T_183393, T_183333)
    node T_183395 = or(T_183394, T_183334)
    node T_183396 = or(T_183395, T_183335)
    node T_183397 = or(T_183396, T_183336)
    node T_183398 = or(T_183397, T_183337)
    node T_183399 = or(T_183398, T_183338)
    node T_183400 = or(T_183399, T_183339)
    node T_183401 = or(T_183400, T_183340)
    node T_183402 = or(T_183401, T_183341)
    wire T_183403 : UInt<1>
    node T_183404 = bits(T_183402, 0, 0)
    T_183403 := T_183404
    node T_183405 = eq(T_183403, UInt<1>(0))
    node T_183406 = and(req.valid, resp.valid)
    node T_183407 = and(T_183406, T_183405)
    wire T_183408 : {history : UInt<7>, value : UInt<2>}
    node T_183409 = bits(req.bits.addr, 8, 2)
    node T_183410 = xor(T_183409, T_183279)
    infer accessor T_183411 = T_183278[T_183410]
    T_183408.value := T_183411
    T_183408.history := T_183279
    node T_183412 = bit(T_183408.value, 0)
    when T_183407 : 
      node T_183413 = bits(T_183279, 6, 1)
      node T_183414 = cat(T_183412, T_183413)
      T_183279 := T_183414
    
    node T_183415 = and(bht_update.valid, bht_update.bits.prediction.valid)
    when T_183415 : 
      node T_183416 = bits(bht_update.bits.pc, 8, 2)
      node T_183417 = xor(T_183416, bht_update.bits.prediction.bits.bht.history)
      infer accessor T_183418 = T_183278[T_183417]
      node T_183419 = bit(bht_update.bits.prediction.bits.bht.value, 1)
      node T_183420 = bit(bht_update.bits.prediction.bits.bht.value, 0)
      node T_183421 = and(T_183419, T_183420)
      node T_183422 = bit(bht_update.bits.prediction.bits.bht.value, 1)
      node T_183423 = bit(bht_update.bits.prediction.bits.bht.value, 0)
      node T_183424 = or(T_183422, T_183423)
      node T_183425 = and(T_183424, bht_update.bits.taken)
      node T_183426 = or(T_183421, T_183425)
      node T_183427 = cat(bht_update.bits.taken, T_183426)
      T_183418 := T_183427
      when bht_update.bits.mispredict : 
        node T_183428 = bits(bht_update.bits.prediction.bits.bht.history, 6, 1)
        node T_183429 = cat(bht_update.bits.taken, T_183428)
        T_183279 := T_183429
    
    
    node T_183430 = bit(T_183408.value, 0)
    node T_183431 = eq(T_183430, UInt<1>(0))
    node T_183432 = and(T_183431, T_183405)
    when T_183432 : resp.bits.taken := UInt<1>(0)
    resp.bits.bht := T_183408
    reg T_183433 : UInt<2>, clk, reset
    onreset T_183433 := UInt<2>(0)
    reg T_183434 : UInt<1>, clk, reset
    onreset T_183434 := UInt<1>(0)
    wire T_183435 : UInt[2]
    T_183435[0] := T_183436
    T_183435[1] := T_183437
    reg T_183438 : UInt[2], clk, reset
    wire T_183439 : UInt<1>
    T_183439 := UInt<1>(0)
    when hits[0] : T_183439 := useRAS[0]
    wire T_183440 : UInt<1>
    T_183440 := UInt<1>(0)
    when hits[1] : T_183440 := useRAS[1]
    wire T_183441 : UInt<1>
    T_183441 := UInt<1>(0)
    when hits[2] : T_183441 := useRAS[2]
    wire T_183442 : UInt<1>
    T_183442 := UInt<1>(0)
    when hits[3] : T_183442 := useRAS[3]
    wire T_183443 : UInt<1>
    T_183443 := UInt<1>(0)
    when hits[4] : T_183443 := useRAS[4]
    wire T_183444 : UInt<1>
    T_183444 := UInt<1>(0)
    when hits[5] : T_183444 := useRAS[5]
    wire T_183445 : UInt<1>
    T_183445 := UInt<1>(0)
    when hits[6] : T_183445 := useRAS[6]
    wire T_183446 : UInt<1>
    T_183446 := UInt<1>(0)
    when hits[7] : T_183446 := useRAS[7]
    wire T_183447 : UInt<1>
    T_183447 := UInt<1>(0)
    when hits[8] : T_183447 := useRAS[8]
    wire T_183448 : UInt<1>
    T_183448 := UInt<1>(0)
    when hits[9] : T_183448 := useRAS[9]
    wire T_183449 : UInt<1>
    T_183449 := UInt<1>(0)
    when hits[10] : T_183449 := useRAS[10]
    wire T_183450 : UInt<1>
    T_183450 := UInt<1>(0)
    when hits[11] : T_183450 := useRAS[11]
    wire T_183451 : UInt<1>
    T_183451 := UInt<1>(0)
    when hits[12] : T_183451 := useRAS[12]
    wire T_183452 : UInt<1>
    T_183452 := UInt<1>(0)
    when hits[13] : T_183452 := useRAS[13]
    wire T_183453 : UInt<1>
    T_183453 := UInt<1>(0)
    when hits[14] : T_183453 := useRAS[14]
    wire T_183454 : UInt<1>
    T_183454 := UInt<1>(0)
    when hits[15] : T_183454 := useRAS[15]
    wire T_183455 : UInt<1>
    T_183455 := UInt<1>(0)
    when hits[16] : T_183455 := useRAS[16]
    wire T_183456 : UInt<1>
    T_183456 := UInt<1>(0)
    when hits[17] : T_183456 := useRAS[17]
    wire T_183457 : UInt<1>
    T_183457 := UInt<1>(0)
    when hits[18] : T_183457 := useRAS[18]
    wire T_183458 : UInt<1>
    T_183458 := UInt<1>(0)
    when hits[19] : T_183458 := useRAS[19]
    wire T_183459 : UInt<1>
    T_183459 := UInt<1>(0)
    when hits[20] : T_183459 := useRAS[20]
    wire T_183460 : UInt<1>
    T_183460 := UInt<1>(0)
    when hits[21] : T_183460 := useRAS[21]
    wire T_183461 : UInt<1>
    T_183461 := UInt<1>(0)
    when hits[22] : T_183461 := useRAS[22]
    wire T_183462 : UInt<1>
    T_183462 := UInt<1>(0)
    when hits[23] : T_183462 := useRAS[23]
    wire T_183463 : UInt<1>
    T_183463 := UInt<1>(0)
    when hits[24] : T_183463 := useRAS[24]
    wire T_183464 : UInt<1>
    T_183464 := UInt<1>(0)
    when hits[25] : T_183464 := useRAS[25]
    wire T_183465 : UInt<1>
    T_183465 := UInt<1>(0)
    when hits[26] : T_183465 := useRAS[26]
    wire T_183466 : UInt<1>
    T_183466 := UInt<1>(0)
    when hits[27] : T_183466 := useRAS[27]
    wire T_183467 : UInt<1>
    T_183467 := UInt<1>(0)
    when hits[28] : T_183467 := useRAS[28]
    wire T_183468 : UInt<1>
    T_183468 := UInt<1>(0)
    when hits[29] : T_183468 := useRAS[29]
    wire T_183469 : UInt<1>
    T_183469 := UInt<1>(0)
    when hits[30] : T_183469 := useRAS[30]
    wire T_183470 : UInt<1>
    T_183470 := UInt<1>(0)
    when hits[31] : T_183470 := useRAS[31]
    wire T_183471 : UInt<1>
    T_183471 := UInt<1>(0)
    when hits[32] : T_183471 := useRAS[32]
    wire T_183472 : UInt<1>
    T_183472 := UInt<1>(0)
    when hits[33] : T_183472 := useRAS[33]
    wire T_183473 : UInt<1>
    T_183473 := UInt<1>(0)
    when hits[34] : T_183473 := useRAS[34]
    wire T_183474 : UInt<1>
    T_183474 := UInt<1>(0)
    when hits[35] : T_183474 := useRAS[35]
    wire T_183475 : UInt<1>
    T_183475 := UInt<1>(0)
    when hits[36] : T_183475 := useRAS[36]
    wire T_183476 : UInt<1>
    T_183476 := UInt<1>(0)
    when hits[37] : T_183476 := useRAS[37]
    wire T_183477 : UInt<1>
    T_183477 := UInt<1>(0)
    when hits[38] : T_183477 := useRAS[38]
    wire T_183478 : UInt<1>
    T_183478 := UInt<1>(0)
    when hits[39] : T_183478 := useRAS[39]
    wire T_183479 : UInt<1>
    T_183479 := UInt<1>(0)
    when hits[40] : T_183479 := useRAS[40]
    wire T_183480 : UInt<1>
    T_183480 := UInt<1>(0)
    when hits[41] : T_183480 := useRAS[41]
    wire T_183481 : UInt<1>
    T_183481 := UInt<1>(0)
    when hits[42] : T_183481 := useRAS[42]
    wire T_183482 : UInt<1>
    T_183482 := UInt<1>(0)
    when hits[43] : T_183482 := useRAS[43]
    wire T_183483 : UInt<1>
    T_183483 := UInt<1>(0)
    when hits[44] : T_183483 := useRAS[44]
    wire T_183484 : UInt<1>
    T_183484 := UInt<1>(0)
    when hits[45] : T_183484 := useRAS[45]
    wire T_183485 : UInt<1>
    T_183485 := UInt<1>(0)
    when hits[46] : T_183485 := useRAS[46]
    wire T_183486 : UInt<1>
    T_183486 := UInt<1>(0)
    when hits[47] : T_183486 := useRAS[47]
    wire T_183487 : UInt<1>
    T_183487 := UInt<1>(0)
    when hits[48] : T_183487 := useRAS[48]
    wire T_183488 : UInt<1>
    T_183488 := UInt<1>(0)
    when hits[49] : T_183488 := useRAS[49]
    wire T_183489 : UInt<1>
    T_183489 := UInt<1>(0)
    when hits[50] : T_183489 := useRAS[50]
    wire T_183490 : UInt<1>
    T_183490 := UInt<1>(0)
    when hits[51] : T_183490 := useRAS[51]
    wire T_183491 : UInt<1>
    T_183491 := UInt<1>(0)
    when hits[52] : T_183491 := useRAS[52]
    wire T_183492 : UInt<1>
    T_183492 := UInt<1>(0)
    when hits[53] : T_183492 := useRAS[53]
    wire T_183493 : UInt<1>
    T_183493 := UInt<1>(0)
    when hits[54] : T_183493 := useRAS[54]
    wire T_183494 : UInt<1>
    T_183494 := UInt<1>(0)
    when hits[55] : T_183494 := useRAS[55]
    wire T_183495 : UInt<1>
    T_183495 := UInt<1>(0)
    when hits[56] : T_183495 := useRAS[56]
    wire T_183496 : UInt<1>
    T_183496 := UInt<1>(0)
    when hits[57] : T_183496 := useRAS[57]
    wire T_183497 : UInt<1>
    T_183497 := UInt<1>(0)
    when hits[58] : T_183497 := useRAS[58]
    wire T_183498 : UInt<1>
    T_183498 := UInt<1>(0)
    when hits[59] : T_183498 := useRAS[59]
    wire T_183499 : UInt<1>
    T_183499 := UInt<1>(0)
    when hits[60] : T_183499 := useRAS[60]
    wire T_183500 : UInt<1>
    T_183500 := UInt<1>(0)
    when hits[61] : T_183500 := useRAS[61]
    node T_183501 = or(T_183439, T_183440)
    node T_183502 = or(T_183501, T_183441)
    node T_183503 = or(T_183502, T_183442)
    node T_183504 = or(T_183503, T_183443)
    node T_183505 = or(T_183504, T_183444)
    node T_183506 = or(T_183505, T_183445)
    node T_183507 = or(T_183506, T_183446)
    node T_183508 = or(T_183507, T_183447)
    node T_183509 = or(T_183508, T_183448)
    node T_183510 = or(T_183509, T_183449)
    node T_183511 = or(T_183510, T_183450)
    node T_183512 = or(T_183511, T_183451)
    node T_183513 = or(T_183512, T_183452)
    node T_183514 = or(T_183513, T_183453)
    node T_183515 = or(T_183514, T_183454)
    node T_183516 = or(T_183515, T_183455)
    node T_183517 = or(T_183516, T_183456)
    node T_183518 = or(T_183517, T_183457)
    node T_183519 = or(T_183518, T_183458)
    node T_183520 = or(T_183519, T_183459)
    node T_183521 = or(T_183520, T_183460)
    node T_183522 = or(T_183521, T_183461)
    node T_183523 = or(T_183522, T_183462)
    node T_183524 = or(T_183523, T_183463)
    node T_183525 = or(T_183524, T_183464)
    node T_183526 = or(T_183525, T_183465)
    node T_183527 = or(T_183526, T_183466)
    node T_183528 = or(T_183527, T_183467)
    node T_183529 = or(T_183528, T_183468)
    node T_183530 = or(T_183529, T_183469)
    node T_183531 = or(T_183530, T_183470)
    node T_183532 = or(T_183531, T_183471)
    node T_183533 = or(T_183532, T_183472)
    node T_183534 = or(T_183533, T_183473)
    node T_183535 = or(T_183534, T_183474)
    node T_183536 = or(T_183535, T_183475)
    node T_183537 = or(T_183536, T_183476)
    node T_183538 = or(T_183537, T_183477)
    node T_183539 = or(T_183538, T_183478)
    node T_183540 = or(T_183539, T_183479)
    node T_183541 = or(T_183540, T_183480)
    node T_183542 = or(T_183541, T_183481)
    node T_183543 = or(T_183542, T_183482)
    node T_183544 = or(T_183543, T_183483)
    node T_183545 = or(T_183544, T_183484)
    node T_183546 = or(T_183545, T_183485)
    node T_183547 = or(T_183546, T_183486)
    node T_183548 = or(T_183547, T_183487)
    node T_183549 = or(T_183548, T_183488)
    node T_183550 = or(T_183549, T_183489)
    node T_183551 = or(T_183550, T_183490)
    node T_183552 = or(T_183551, T_183491)
    node T_183553 = or(T_183552, T_183492)
    node T_183554 = or(T_183553, T_183493)
    node T_183555 = or(T_183554, T_183494)
    node T_183556 = or(T_183555, T_183495)
    node T_183557 = or(T_183556, T_183496)
    node T_183558 = or(T_183557, T_183497)
    node T_183559 = or(T_183558, T_183498)
    node T_183560 = or(T_183559, T_183499)
    node T_183561 = or(T_183560, T_183500)
    wire T_183562 : UInt<1>
    node T_183563 = bits(T_183561, 0, 0)
    T_183562 := T_183563
    
    
    node T_183564 = eq(T_183433, UInt<1>(0))
    node T_183565 = eq(T_183564, UInt<1>(0))
    node T_183566 = and(T_183565, T_183562)
    when T_183566 : 
      infer accessor T_183567 = T_183438[T_183434]
      resp.bits.target := T_183567
    when ras_update.valid : when ras_update.bits.isCall : 
        
        node T_183576 = lt(T_183433, UInt<2>(2))
        when T_183576 : 
          node T_183577 = addw(T_183433, UInt<1>(1))
          T_183433 := T_183577
        node T_183578 = lt(T_183434, UInt<1>(1))
        node T_183579 = or(UInt<1>(1), T_183578)
        node T_183580 = addw(T_183434, UInt<1>(1))
        wire T_183581 : UInt<1>
        T_183581 := UInt<1>(0)
        when T_183579 : T_183581 := T_183580
        infer accessor T_183582 = T_183438[T_183581]
        T_183582 := ras_update.bits.returnAddr
        T_183434 := T_183581
        when T_183562 : resp.bits.target := ras_update.bits.returnAddr
      else : 
        
        node T_183568 = and(ras_update.bits.isReturn, ras_update.bits.prediction.valid)
        when T_183568 : 
          
          node T_183569 = eq(T_183433, UInt<1>(0))
          node T_183570 = eq(T_183569, UInt<1>(0))
          when T_183570 : 
            node T_183571 = subw(T_183433, UInt<1>(1))
            T_183433 := T_183571
            node T_183572 = gt(T_183434, UInt<1>(0))
            node T_183573 = or(UInt<1>(1), T_183572)
            node T_183574 = subw(T_183434, UInt<1>(1))
            wire T_183575 : UInt<1>
            T_183575 := UInt<1>(1)
            when T_183573 : T_183575 := T_183574
            T_183434 := T_183575
    when invalidate : T_183433 := UInt<1>(0)
