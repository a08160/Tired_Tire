import 'package:flutter/material.dart';
import 'package:flutter_application_1/pages/welcome_page.dart';
import 'dart:async';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Timer(const Duration(seconds: 3), () {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => WelcomePage()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF2C2B34), // 배경색 #2c2b34
      body: Center(
        child: Image.asset(
          'assets/splash/logo.png',
          width: 400, // 크기 조절 가능
        ),
      ),
    );
  }
}
