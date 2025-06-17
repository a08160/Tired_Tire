import 'package:flutter/material.dart';
import 'signup_page.dart';
import 'login_page.dart';

class WelcomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF2C2B34), // 전체 배경 #2C2B34
      body: SafeArea(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Spacer(),

            // 자동차 이미지 (왼쪽 조금 자르고 전체 너비 채우기)
            ClipRect(
              child: Align(
                alignment: Alignment.centerLeft, // 왼쪽을 기준으로 잘라냄
                widthFactor: 1.2, // 1보다 크면 왼쪽이 잘림
                child: Image.asset(
                  'assets/car.png',
                  width: double.infinity, // 가로 전체 채움
                  fit: BoxFit.cover, // 잘 맞추도록 자름
                ),
              ),
            ),

            SizedBox(height: 30),

            Text(
              'Tired Tire',
              style: TextStyle(
                color: Color(0xFFFFFFFF), // 흰색 글자
                fontSize: 28,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 10),
            Text(
              '당신의 타이어, 지금 Tired 하진 않나요?',
              style: TextStyle(
                color: Color(0xFFBFBFBF), // 연한 회색 글자
                fontSize: 14,
              ),
            ),
            SizedBox(height: 40),

            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 40),
              child: Column(
                children: [
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => SignUpPage()),
                      );
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: Colors.black,
                      minimumSize: Size(double.infinity, 48),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                    child: Text('회원가입'),
                  ),
                  SizedBox(height: 10),
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => LoginPage()),
                      );
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white24,
                      foregroundColor: Colors.white,
                      minimumSize: Size(double.infinity, 48),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                    child: Text('로그인'),
                  ),
                ],
              ),
            ),

            Spacer(),
          ],
        ),
      ),
    );
  }
}
