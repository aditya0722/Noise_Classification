import React from "react";

const Footer = () => {
  return (
    <footer className="w-full bg-blue-900 text-white py-6 px-4 mt-16 animate-fade-in">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center">
        <p className="text-sm text-center md:text-left">
          © {new Date().getFullYear()} Noise Classification Project. All rights reserved.
        </p>
        <p className="text-sm text-center md:text-right mt-2 md:mt-0">
          Designed & Developed By Aditya Sharma, Agimik Marak, Madhusmita Sen
        </p>
      </div>
    </footer>
  );
};

export default Footer;
