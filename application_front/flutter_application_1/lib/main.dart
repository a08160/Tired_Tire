// 실행하기 전에 터미널에 아래 명령어 입력하기
// uvicorn inference_server:app --host 0.0.0.0 --port 8000 --reload
// uvicorn crack_inference_server:app --host 0.0.0.0 --port 8001 --reload
// image_picker_modal.dart & crack_inference_server.py & crack_image_modal 에서 ip주소 변경(ipconfig)

import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'pages/welcome_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized(); // Flutter와 Firebase 연결을 위해 필요
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tired Tire',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: Colors.white, // 배경색 흰색
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.white, // 앱바 배경 흰색
          elevation: 0,
        ),
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.white,
        ).copyWith(background: Colors.white),
      ),
      home: WelcomePage(),
    );
  }
}
