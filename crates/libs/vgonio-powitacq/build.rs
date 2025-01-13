fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("cxx/powitacq.cc")
        .std("c++11")
        .compile("vgonio-powitacq");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cxx/powitacq.cc");
    println!("cargo:rerun-if-changed=cxx/powitacq.h");
}
