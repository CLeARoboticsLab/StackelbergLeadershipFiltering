
mc_local_folder = "mc_data"
mc_server_folder = "mc_data_server"


function get_final_lq_paths()
    #x20 on server
    data_folder = "FINAL_lq_mc20_L2_8_7_2_41"
    silq_data_file = "lq_silq_mc20_L2_th0.015_ss0.01_M50.jld"
    lf_data_file = "lq_lf_mc20_L2_th0.015_ss0.01_M50.jld"
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
    data_folder = "lq_mc2_L2_8_6_19_33"
    silq_data_file = "lq_silq_mc2_L2_th0.015_ss0.01_M50.jld"
    lf_data_file = "lq_lf_mc2_L2_th0.015_ss0.01_M50.jld"

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

# data_folder, silq_data_file, lf_data_file = get_lnq_paths()

mc_folder = mc_server_folder
data_folder, silq_data_file, lf_data_file = get_final_lq_paths()
