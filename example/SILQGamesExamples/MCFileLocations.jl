
mc_local_folder = "mc_data"
mc_server_folder = "mc_data_server"


function get_final_lq_paths_p1()
    #x20 on server
    data_folder = "lq_mc100_L1_8_8_13_14"
    # "lq_mc20_L1_8_8_12_14"
    #"FINAL_lq_mc20_L2_8_7_2_41"
    silq_data_file = "lq_silq_mc100_L1_th0.015_ss0.01_M50.jld"#"lq_silq_mc20_L1_th0.015_ss0.01_M50.jld"
    lf_data_file = "lq_lf_mc100_L1_th0.015_ss0.01_M50.jld"#"lq_lf_mc20_L1_th0.015_ss0.01_M50.jld"


    # # data_folder = "lq_mc20_L1_8_7_16_48"
    # data_folder = "lq_mc20_L1_8_7_22_49"
    # silq_data_file = "lq_silq_mc20_L1_th0.015_ss0.01_M50.jld"
    # lf_data_file = "lq_lf_mc20_L1_th0.001_ss0.01_M50.jld"

    # data_folder = ""

    return data_folder, silq_data_file, lf_data_file
end

function get_final_lq_paths_p2()
    data_folder="lq_mc20_L2_8_8_15_6"
    silq_data_file="lq_silq_mc20_L2_th0.015_ss0.01_M50.jld"
    lf_data_file="lq_lf_mc20_L2_th0.015_ss0.01_M50.jld"
    return data_folder, silq_data_file, lf_data_file
end

# function get_final_uq_paths_p1()
#     data_folder = "uq_mc20_L1_8_8_17_4"
#     silq_data_file="uq_silq_mc20_L1_th0.003_ss0.01_M1000.jld"
#     lf_data_file="uq_lf_mc20_L1_th0.001_ss0.01_M50.jld"


#     # data_folder="uq_mc20_L1_8_8_23_10"
#     # silq_data_file="uq_silq_mc20_L1_th0.003_ss0.01_M1500.jld"
#     # lf_data_file=""

#     data_folder="uq_mc50_L1_8_9_16_5"
#     silq_data_file="uq_silq_mc50_L1_th0.003_ss0.01_M1500.jld"
#     lf_data_file = "uq_lf_mc50_L1_th0.001_ss0.01_M50.jld"

#     return data_folder, silq_data_file, lf_data_file
# end

function get_final_uq_paths_p1()
    data_folder = "uq_mc20_L1_8_7_13_56"
    silq_data_file = "uq_silq_mc20_L1_th0.003_ss0.01_M1000.jld"
    lf_data_file = "uq_lf_mc20_L1_th0.001_ss0.01_M50.jld"

    data_folder = "uq_mc20_L1_8_9_2_39"
    silq_data_file = "uq_silq_mc20_L1_th0.003_ss0.01_M1500.jld"
    lf_data_file = "uq_lf_mc20_L1_th0.001_ss0.01_M50.jld"

    # data_folder="uq_mc50_L1_8_9_16_5"
    # silq_data_file="uq_silq_mc50_L1_th0.003_ss0.01_M1500.jld"
    # lf_data_file = "uq_lf_mc50_L1_th0.001_ss0.01_M50.jld"

    return data_folder, silq_data_file, lf_data_file
end

function get_final_lnq_paths_p2()
    data_folder="lnq_mc10_L2_8_8_23_39"
    silq_data_file="lnq_silq_mc10_L2_th0.0015_ss0.01_M2500.jld"
    lf_data_file=""

    return data_folder, silq_data_file, lf_data_file
end

function get_final_nonlq_paths_p2()
    data_folder="nonlq_mc10_L2_8_9_2_39"
    silq_data_file="nonlq_silq_mc10_L2_th0.001_ss0.01_M3000.jld"
    lf_data_file="nonlq_lf_mc10_L2_th0.001_ss0.02_M50.jld"

    data_folder="nonlq_mc20_L2_8_31_9_40"
    silq_data_file="nonlq_silq_mc20_L2_th0.001_ss0.01_M3000.jld"
    lf_data_file="nonlq_lf_mc20_L2_th0.001_ss0.02_M50.jld"

    return data_folder, silq_data_file, lf_data_file
end

function get_lq_paths()
    data_folder = "lq_mc2_L2_8_3_17_41"
    silq_data_file = "lq_silq_mc2_L2_th0.015_ss0.01_M50.jld"
    lf_data_file = "lq_lf_mc2_L2_th0.015_ss0.01_M50.jld"


    data_folder = "lq_mc2_L2_8_6_16_1"
    silq_data_file = "lq_silq_mc2_L2_th0.015_ss0.01_M50.jld"
    lf_data_file = ""

    # new
    data_folder = "lq_mc2_L1_8_8_11_22" #"lq_mc20_L1_8_7_21_44"
    silq_data_file = "lq_silq_mc2_L1_th0.015_ss0.01_M50.jld"
    lf_data_file = "lq_lf_mc2_L1_th0.015_ss0.01_M50.jld"

    data_folder = "lq_mc20_L1_8_8_11_29"
    silq_data_file = "lq_silq_mc20_L1_th0.015_ss0.01_M50.jld"
    lf_data_file = "lq_lf_mc20_L1_th0.015_ss0.01_M50.jld"

    # # old
    # data_folder = "lq_mc2_L2_8_6_16_36"
    # silq_data_file = "lq_silq_mc2_L2_th0.015_ss0.01_M50.jld"
    # lf_data_file = "lq_lf_mc2_L2_th0.015_ss0.01_M50.jld"

    return data_folder, silq_data_file, lf_data_file
end

function get_uq_paths()
    data_folder = "uq_mc2_L1_8_6_18_29"
    silq_data_file = "uq_silq_mc2_L1_th0.003_ss0.01_M1000.jld"
    lf_data_file = ""
    return data_folder, silq_data_file, lf_data_file
end

# data_folder
function get_lnq_paths()
    data_folder="TWOIDENTICAL_lnq_mc2_L2_8_6_14_48"
    silq_data_file = "lnq_silq_mc2_L2_th0.001_ss0.01_M2000.jld"
    lf_data_file = ""

    data_folder="lnq_mc1_L2_8_6_20_11"
    silq_data_file="lnq_silq_mc1_L2_th0.001_ss0.01_M2000.jld"
    lf_data_file="lnq_lf_mc1_L2_th0.001_ss0.02_M50.jld"

    data_folder="lnq_mc3_L2_8_6_21_34"
    silq_data_file="lnq_silq_mc3_L2_th0.0015_ss0.01_M2500.jld"
    lf_data_file="lnq_lf_mc3_L2_th0.001_ss0.02_M50.jld"

    # data_folder="lnq_mc3_L2_8_6_23_27"
    # silq_data_file="lnq_silq_mc3_L2_th0.00125_ss0.01_M2500.jld"

    return data_folder, silq_data_file, lf_data_file
end


# x20 UQ simulations
data_folder = "uq_mc20_L1_8_5_12_4"
lf_data_file = "uq_lf_mc20_L1_th0.001_ss0.01_M50.jld"
silq_data_file = "uq_silq_mc20_L1_th0.004_ss0.01_M1000.jld"

mc_folder = mc_local_folder
data_folder, silq_data_file, lf_data_file = get_lq_paths()

#mc_folder = mc_server_folder
data_folder, silq_data_file, lf_data_file = get_final_nonlq_paths_p2() 
data_folder, silq_data_file, lf_data_file = get_final_lq_paths_p1()
